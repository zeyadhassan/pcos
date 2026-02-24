"""Hybrid Retrieval Planner (§9).

Retrieves the right memories for the right task:
1. Intent classification
2. Entity linking (resolved to KG node UUIDs)
3. Graph retrieval (structured facts, dependencies, constraints)
4. Episodic retrieval (similar past situations)
5. Temporal filtering (date-window + recency)
6. Policy filtering
7. Context assembly + relevance ranking
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from percos.llm import LLMClient
from percos.logging import get_logger
from percos.models.enums import IntentType
from percos.models.events import (
    CommittedFact,
    ContextBundle,
    EpisodicEntry,
    PolicyEntry,
    WorkingMemory,
)
from percos.stores.episodic_store import EpisodicStore
from percos.stores.graph import KnowledgeGraph
from percos.stores.procedural_policy_stores import PolicyStore
from percos.stores.semantic_store import SemanticStore

log = get_logger("retrieval")

INTENT_SYSTEM_PROMPT = """\
Classify the user's intent into one of: recall, planning, execution, reflection.
Also extract any entity names, dates, or project references mentioned.

Respond with JSON:
{
  "intent": "recall" | "planning" | "execution" | "reflection",
  "entities": ["entity1", "entity2"],
  "date_references": ["2025-03-15"],
  "keywords": ["keyword1"]
}
"""

# ── Helpers ─────────────────────────────────────────────

def _get_entity_types() -> list[str]:
    """Return entity type names from the active domain schema.

    Falls back to a hardcoded list for backward compatibility.
    """
    try:
        from percos.schema import get_domain_schema
        schema = get_domain_schema()
        return schema.get_entity_type_names()
    except Exception:
        return ["entity"]


def _parse_date(s: str) -> datetime | None:
    """Best-effort ISO-date parse (date or datetime)."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _compute_relevance(
    fact_row,
    *,
    entity_match: bool = False,
    keyword_match: bool = False,
    date_range: tuple[datetime, datetime] | None = None,
    ranking_strategy: str = "recency_confidence",
) -> float:
    """Score a semantic fact row for context relevance (0-1).

    *ranking_strategy* (from evolution config) controls the weighting:
      - ``recency_confidence`` (default): balanced recency + confidence
      - ``confidence_first``: strongly favour high-confidence facts
      - ``recency_first``: strongly favour recent facts
      - ``entity_match_first``: strongly favour entity/keyword matches
    """
    score = 0.0

    # ── Weight profiles per strategy ──
    weights = {
        "recency_confidence": {"conf": 0.30, "entity": 0.25, "keyword": 0.15, "recency": 0.15, "temporal": 0.15},
        "confidence_first":   {"conf": 0.50, "entity": 0.15, "keyword": 0.10, "recency": 0.10, "temporal": 0.15},
        "recency_first":      {"conf": 0.15, "entity": 0.15, "keyword": 0.10, "recency": 0.45, "temporal": 0.15},
        "entity_match_first": {"conf": 0.15, "entity": 0.40, "keyword": 0.20, "recency": 0.10, "temporal": 0.15},
    }
    w = weights.get(ranking_strategy, weights["recency_confidence"])

    # Confidence boost
    conf_map = {"high": 1.0, "medium": 0.65, "low": 0.2}
    score += w["conf"] * conf_map.get(getattr(fact_row, "confidence", "medium"), 0.3)

    # Entity / keyword match
    if entity_match:
        score += w["entity"]
    if keyword_match:
        score += w["keyword"]

    # Recency (exponential decay over 180 days)
    created = getattr(fact_row, "created_at", None)
    if created:
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(tz=timezone.utc) - created).days
        recency = max(0.0, 1.0 - age_days / 180)
        score += w["recency"] * recency

    # Temporal overlap with queried date window
    if date_range:
        valid_from = getattr(fact_row, "valid_from", None)
        valid_to = getattr(fact_row, "valid_to", None) or datetime.now(tz=timezone.utc)
        if valid_from:
            if valid_from.tzinfo is None:
                valid_from = valid_from.replace(tzinfo=timezone.utc)
            if valid_to and valid_to.tzinfo is None:
                valid_to = valid_to.replace(tzinfo=timezone.utc)
            dr_start, dr_end = date_range
            if valid_from <= dr_end and valid_to >= dr_start:
                score += w["temporal"]

    return min(score, 1.0)


class RetrievalPlanner:
    """Assembles the optimal context bundle for a given query/task."""

    def __init__(
        self,
        llm: LLMClient,
        semantic_store: SemanticStore,
        episodic_store: EpisodicStore,
        policy_store: PolicyStore,
        knowledge_graph: KnowledgeGraph,
        evolution_sandbox=None,
    ):
        self._llm = llm
        self._semantic = semantic_store
        self._episodic = episodic_store
        self._policy = policy_store
        self._graph = knowledge_graph
        self._evolution = evolution_sandbox

    # ── Public API ──────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        working_memory: WorkingMemory | None = None,
        *,
        task_type: str | None = None,
        requester_clearance: str = "internal",
    ) -> ContextBundle:
        """Build a context bundle for the query.

        Core interface: query_world_model(query, task_type) -> context_bundle

        Args:
            query: natural-language query.
            working_memory: current session state.
            task_type: optional caller-specified intent (recall/planning/execution/reflection).
                       If omitted the intent is auto-classified via LLM.
            requester_clearance: sensitivity clearance level of the requester
                (public/internal/private/secret). Facts with higher sensitivity
                are filtered out (§9.2 step 6, §11.2).
        """
        # 1. Intent classification + entity extraction
        intent_info = await self._classify_intent(query)
        if task_type:
            try:
                intent = IntentType(task_type)
            except ValueError:
                intent = IntentType(intent_info.get("intent", "recall"))
        else:
            intent = IntentType(intent_info.get("intent", "recall"))

        entities = intent_info.get("entities", [])
        keywords = intent_info.get("keywords", [])
        date_refs = intent_info.get("date_references", [])

        log.info("retrieval_start", intent=intent.value, entities=entities, dates=date_refs)

        # ── Load evolution config ───────────────────────
        retrieval_heuristic = "hybrid"
        ranking_strategy = "recency_confidence"
        if self._evolution:
            try:
                evo_config = await self._evolution.get_active_config()
                retrieval_heuristic = evo_config.get("retrieval_heuristic", "hybrid")
                ranking_strategy = evo_config.get("ranking_strategy", "recency_confidence")
            except Exception:
                pass

        # ── 1b. Parse temporal window ───────────────────
        date_range = self._build_date_range(date_refs)

        # ── 2. Entity linking (resolved to KG node UUIDs) ──
        resolved_entities = self._resolve_entities(entities)

        # ── 3. Graph retrieval ──────────────────────────
        graph_context: dict = {}
        for entity_name, node_ids in resolved_entities.items():
            for nid in node_ids:
                neighborhood = self._graph.neighbors(nid, depth=2)
                if neighborhood.get("nodes"):
                    graph_context[entity_name] = neighborhood
                    break

        # Also do the old-style name matching for entities that didn't resolve
        for entity_name in entities:
            if entity_name in graph_context:
                continue
            for etype in _get_entity_types():
                nodes = self._graph.find_by_type(etype)
                for node in nodes:
                    node_name = node.get("data", {}).get("name", "").lower()
                    if entity_name.lower() in node_name or node_name in entity_name.lower():
                        neighborhood = self._graph.neighbors(node["id"], depth=2)
                        graph_context[entity_name] = neighborhood
                        break

        # Collect the set of resolved fact-ids for relevance scoring
        resolved_fact_ids: set[str] = set()
        for nids in resolved_entities.values():
            resolved_fact_ids.update(nids)

        # ── 4. Semantic fact retrieval ──────────────────
        raw_facts: list[tuple] = []  # (row, entity_match, keyword_match)
        seen_ids: set[str] = set()

        for entity_name in entities:
            facts = await self._semantic.search_facts(entity_name)
            for f in facts:
                if f.id not in seen_ids:
                    seen_ids.add(f.id)
                    raw_facts.append((f, True, False))

        for kw in keywords:
            facts = await self._semantic.search_facts(kw)
            for f in facts:
                if f.id not in seen_ids:
                    seen_ids.add(f.id)
                    raw_facts.append((f, False, True))
                elif f.id in seen_ids:
                    # upgrade keyword flag on existing entry
                    for i, (rf, em, km) in enumerate(raw_facts):
                        if rf.id == f.id:
                            raw_facts[i] = (rf, em, True)
                            break

        # ── 5. Temporal filtering ───────────────────────
        if date_range:
            raw_facts = self._apply_temporal_filter(raw_facts, date_range)

        # ── 5b. Sensitivity filtering (§9.2 step 6, §11.2 – GAP-C4) ──
        from percos.engine.security import check_sensitivity_access
        raw_facts = [
            (f, em, km) for f, em, km in raw_facts
            if check_sensitivity_access(
                getattr(f, "sensitivity", "internal"), requester_clearance
            )
        ]

        # ── 6. Relevance ranking ────────────────────────
        scored: list[tuple[float, object, bool, bool]] = []
        relevance_scores: dict[str, float] = {}
        for f, em, km in raw_facts:
            score = _compute_relevance(
                f, entity_match=em, keyword_match=km,
                date_range=date_range, ranking_strategy=ranking_strategy,
            )
            scored.append((score, f, em, km))
            relevance_scores[str(f.id)] = round(score, 4)
        scored.sort(key=lambda t: t[0], reverse=True)

        semantic_facts = []
        for _score, f, _em, _km in scored:
            semantic_facts.append(CommittedFact(
                id=f.id,
                candidate_id=f.candidate_id,
                entity_type=f.entity_type,
                entity_data=f.entity_data,
                fact_type=f.fact_type,
                confidence=f.confidence,
                scope=f.scope,
                sensitivity=f.sensitivity,
                source=f.source,
                provenance_chain=f.provenance_chain or [],
            ))

        # ── 7. Episodic retrieval (vector search) ──────
        # retrieval_heuristic controls depth: hybrid (default) uses both
        # semantic + episodic, semantic_only skips episodic, episodic_heavy
        # doubles the episodic result count.
        episodic_entries: list[EpisodicEntry] = []
        if retrieval_heuristic != "semantic_only":
            ep_count = 20 if retrieval_heuristic == "episodic_heavy" else 10
            similar = self._episodic.search_similar(query, n_results=ep_count)
            for item in similar:
                # Sensitivity filter for episodic entries (§9.2 – GAP-C4)
                ep_sensitivity = item.get("metadata", {}).get("sensitivity", "internal")
                if not check_sensitivity_access(ep_sensitivity, requester_clearance):
                    continue

                ep = EpisodicEntry(
                    id=item["id"],
                    event_id=item["metadata"].get("event_id", ""),
                    content=item.get("document", ""),
                    metadata_extra=item.get("metadata", {}),
                )
                # Temporal filter on episodic entries
                if date_range:
                    ts_str = item.get("metadata", {}).get("timestamp")
                    if ts_str:
                        ts = _parse_date(ts_str)
                        if ts and not (date_range[0] <= ts <= date_range[1]):
                            continue
                episodic_entries.append(ep)

        # ── 8. Policy filtering ─────────────────────────
        policies: list[PolicyEntry] = []
        active_policies = await self._policy.get_active_policies()
        for p in active_policies:
            policies.append(PolicyEntry(
                id=p.id,
                name=p.name,
                rule=p.rule,
                effect=p.effect,
                priority=p.priority,
                scope=p.scope,
                active=p.active,
            ))

        # ── 9. Assemble context bundle ──────────────────
        bundle = ContextBundle(
            query=query,
            semantic_facts=semantic_facts,
            episodic_entries=episodic_entries,
            policies=policies,
            working_memory=working_memory or WorkingMemory(),
            graph_context=graph_context,
            relevance_scores=relevance_scores,
        )

        log.info(
            "retrieval_complete",
            semantic_count=len(semantic_facts),
            episodic_count=len(episodic_entries),
            graph_entities=len(graph_context),
        )
        return bundle

    # ── Entity linking ──────────────────────────────────

    def _resolve_entities(self, entity_names: list[str]) -> dict[str, list[str]]:
        """Resolve textual entity names to KG node UUIDs.

        Returns a mapping from entity name → list of matched node IDs.
        Matches via exact name, substring, and alias scanning across all
        ontology types.
        """
        resolved: dict[str, list[str]] = {}
        for ename in entity_names:
            name_lower = ename.lower()
            matched_ids: list[str] = []
            for etype in _get_entity_types():
                for node in self._graph.find_by_type(etype):
                    data = node.get("data", {})
                    node_name = (data.get("name") or "").lower()
                    if not node_name:
                        continue
                    # Exact match
                    if name_lower == node_name:
                        matched_ids.insert(0, node["id"])  # prioritise
                        continue
                    # Substring match (either direction)
                    if name_lower in node_name or node_name in name_lower:
                        matched_ids.append(node["id"])
                        continue
                    # Alias / alternative names
                    aliases = data.get("aliases", [])
                    if isinstance(aliases, list):
                        for alias in aliases:
                            if name_lower == str(alias).lower():
                                matched_ids.append(node["id"])
                                break
            if matched_ids:
                # Deduplicate while preserving order
                seen: set[str] = set()
                deduped: list[str] = []
                for mid in matched_ids:
                    if mid not in seen:
                        seen.add(mid)
                        deduped.append(mid)
                resolved[ename] = deduped
        return resolved

    # ── Temporal helpers ────────────────────────────────

    @staticmethod
    def _build_date_range(date_refs: list[str]) -> tuple[datetime, datetime] | None:
        """Convert date references into a (start, end) temporal window.

        Behaviour:
        - 0 date refs → None (no filter)
        - 1 date ref  → window = that day ± 1 day
        - 2+ date refs → (earliest, latest)
        """
        parsed: list[datetime] = []
        for dref in date_refs:
            dt = _parse_date(dref)
            if dt:
                parsed.append(dt)
        if not parsed:
            return None
        if len(parsed) == 1:
            d = parsed[0]
            return (d - timedelta(days=1), d + timedelta(days=1))
        parsed.sort()
        return (parsed[0], parsed[-1])

    @staticmethod
    def _apply_temporal_filter(
        raw_facts: list[tuple],
        date_range: tuple[datetime, datetime],
    ) -> list[tuple]:
        """Keep only facts whose validity window overlaps the date range."""
        dr_start, dr_end = date_range
        filtered: list[tuple] = []
        for entry in raw_facts:
            f = entry[0]
            valid_from = getattr(f, "valid_from", None)
            valid_to = getattr(f, "valid_to", None) or datetime.now(tz=timezone.utc)
            if valid_from:
                if valid_from.tzinfo is None:
                    valid_from = valid_from.replace(tzinfo=timezone.utc)
                if valid_to and valid_to.tzinfo is None:
                    valid_to = valid_to.replace(tzinfo=timezone.utc)
                if valid_from <= dr_end and valid_to >= dr_start:
                    filtered.append(entry)
            else:
                filtered.append(entry)  # keep if no temporal info
        return filtered

    # ── Intent classification ───────────────────────────

    async def _classify_intent(self, query: str) -> dict:
        """Classify the intent of a query using LLM."""
        try:
            result = await self._llm.extract_structured(INTENT_SYSTEM_PROMPT, query)
            return result if isinstance(result, dict) else {"intent": "recall"}
        except Exception as exc:
            log.warning("intent_classification_failed", error=str(exc))
            return {"intent": "recall", "entities": [], "keywords": []}
