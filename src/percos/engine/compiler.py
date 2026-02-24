"""Memory Compiler – converts raw events into structured candidate knowledge (§7).

Pipeline:
  1. Parse event
  2. Extract entities / relations / preferences / commitments (LLM)
  3. Link to ontology nodes
  4. Assign confidence + provenance
  5. Detect conflicts with existing graph
  6. Route candidate to: auto_accept | needs_verification | needs_user_confirm | quarantine
"""

from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from percos.llm import LLMClient
from percos.logging import get_logger
from percos.models.enums import CandidateRouting, Confidence, FactType, Sensitivity
from percos.models.events import CandidateFact, CommittedFact
from percos.stores.semantic_store import SemanticStore
from percos.stores.tables import CandidateFactRow, EventRow
from percos.engine.security import meets_confidence_threshold, sanitize_input

log = get_logger("memory_compiler")

# ── Ontology model registry – sourced from the domain schema (§6.1) ──
_ONTOLOGY_MODEL_MAP: dict[str, type] | None = None


def _get_ontology_model_map() -> dict[str, type]:
    """Return entity_type → Pydantic model mapping from the active domain schema."""
    global _ONTOLOGY_MODEL_MAP
    if _ONTOLOGY_MODEL_MAP is not None:
        return _ONTOLOGY_MODEL_MAP
    from percos.schema import get_domain_schema
    schema = get_domain_schema()
    _ONTOLOGY_MODEL_MAP = schema.get_model_map()
    return _ONTOLOGY_MODEL_MAP


def _get_extraction_prompt() -> str:
    """Return the extraction system prompt from the active domain schema.

    Falls back to a default prompt if schema is not available.
    """
    try:
        from percos.schema import get_domain_schema
        schema = get_domain_schema()
        return schema.get_extraction_prompt()
    except Exception:
        return _FALLBACK_EXTRACTION_PROMPT


_FALLBACK_EXTRACTION_PROMPT = """\
You are a knowledge extraction engine.
Given a raw event (conversation, document, etc.),
extract structured knowledge candidates.

For each candidate, output a JSON object with:
- entity_type: string identifying the type of entity
- entity_data: dict with relevant fields (name, description, etc.)
- relation_type: optional string if this establishes a relation
- relation_source: optional name/id of source entity
- relation_target: optional name/id of target entity
- fact_type: one of [observed, derived, hypothesis, policy]
- confidence: one of [high, medium, low]
- scope: string identifying the scope (e.g. global)
- sensitivity: one of [public, internal, private, secret]

Return a JSON object with a "candidates" array.
If no knowledge can be extracted, return {"candidates": []}.
Be precise. Do not hallucinate entities. Only extract what is clearly present or
reasonably inferable from the input.
"""


class MemoryCompiler:
    """Extracts candidate facts from raw events using LLM + ontology linking.

    GAP-H8: Supports dynamic ontology extensions from the evolution pipeline.
    """

    def __init__(
        self,
        session: AsyncSession,
        llm: LLMClient,
        semantic_store: SemanticStore,
        evolution_sandbox=None,
    ):
        self._session = session
        self._llm = llm
        self._semantic = semantic_store
        self._evolution = evolution_sandbox

    async def compile(self, event_id: str) -> list[CandidateFact]:
        """Compile candidate facts from a raw event.

        Core interface: compile_memory(event_id) -> candidate_facts[]
        """
        # 1. Load event
        stmt = select(EventRow).where(EventRow.id == event_id)
        result = await self._session.execute(stmt)
        event_row = result.scalar_one_or_none()
        if not event_row:
            log.warning("event_not_found", event_id=event_id)
            return []

        # 2. LLM extraction
        # Use domain schema extraction prompt; include evolution extensions if available
        extraction_prompt = _get_extraction_prompt()
        if self._evolution:
            try:
                config = await self._evolution.get_active_config()
                extensions = config.get("ontology_extensions", [])
                if extensions:
                    ext_types = []
                    for ext in extensions:
                        ext_types.extend(ext.get("entity_types", []))
                    if ext_types:
                        type_names = [t if isinstance(t, str) else t.get("name", "") for t in ext_types]
                        extraction_prompt += (
                            f"\n\nAdditional custom entity types available: {', '.join(type_names)}"
                        )
            except Exception:
                pass

        user_content = (
            f"Event type: {event_row.event_type}\n"
            f"Source: {event_row.source}\n"
            f"Timestamp: {event_row.timestamp.isoformat()}\n"
            f"Content:\n{event_row.content}"
        )
        try:
            extraction = await self._llm.extract_structured(
                extraction_prompt, user_content
            )
        except Exception as exc:
            log.error("extraction_failed", event_id=event_id, error=str(exc))
            return []

        raw_candidates = extraction.get("candidates", []) if isinstance(extraction, dict) else []

        # 3. Build CandidateFact objects
        candidates: list[CandidateFact] = []
        for raw in raw_candidates:
            # Resolve relation source/target names to existing fact IDs (§7.4 step 3)
            relation_source_id = None
            relation_target_id = None
            relation_source_name = raw.get("relation_source")
            relation_target_name = raw.get("relation_target")
            if relation_source_name:
                resolved = await self._resolve_entity_by_name(relation_source_name)
                if resolved:
                    relation_source_id = UUID(resolved)
            if relation_target_name:
                resolved = await self._resolve_entity_by_name(relation_target_name)
                if resolved:
                    relation_target_id = UUID(resolved)

            candidate = CandidateFact(
                event_id=UUID(event_id),
                entity_type=raw.get("entity_type", ""),
                entity_data=raw.get("entity_data", {}),
                relation_type=raw.get("relation_type"),
                relation_source_id=relation_source_id,
                relation_target_id=relation_target_id,
                fact_type=FactType(raw.get("fact_type", "observed")),
                confidence=Confidence(raw.get("confidence", "medium")),
                scope=raw.get("scope", "global"),
                sensitivity=Sensitivity(raw.get("sensitivity", "internal")),
                raw_extraction=json.dumps(raw),
                provenance=[f"event:{event_id}", f"llm_extraction"],
            )

            # 3b. Entity linking against existing KG nodes (§7.4 step 3 – GAP-M4)
            await self._link_to_existing_entity(candidate)

            # 4. Ontology schema validation (§6.1 – GAP-C6)
            validation_warnings = self._validate_against_ontology(
                candidate.entity_type, candidate.entity_data,
            )
            if validation_warnings:
                candidate.entity_data["_ontology_warnings"] = validation_warnings
                log.info(
                    "ontology_validation_warnings",
                    entity_type=candidate.entity_type,
                    warnings=validation_warnings,
                )

            # 5. Conflict detection
            conflicts = await self._detect_conflicts(candidate)
            candidate.conflicts_with = conflicts

            # 6. Route
            candidate.routing = self._route(candidate)

            candidates.append(candidate)

        # 7. Identity resolution — link candidates to existing entities (GAP-22: §8.4 step 5)
        try:
            from percos.engine.identity_resolution import IdentityResolver
            resolver = IdentityResolver(self._session)
            for c in candidates:
                name = c.entity_data.get("name", "")
                if name and c.entity_type:
                    dupes = await resolver.find_duplicates(entity_type=c.entity_type)
                    for dupe in dupes:
                        # If this candidate's name matches a known duplicate pair, log it
                        if name.lower() in str(dupe).lower():
                            c.entity_data.setdefault("_identity_matches", []).append(dupe)
                            log.info("identity_match_found", candidate=str(c.id),
                                     entity_type=c.entity_type, match=dupe)
                            break
        except Exception:
            log.debug("identity_resolution_skipped")

        # 8. Persist candidate rows
        for c in candidates:
            row = CandidateFactRow(
                id=str(c.id),
                event_id=str(c.event_id),
                entity_type=c.entity_type,
                entity_data=c.entity_data,
                relation_type=c.relation_type,
                relation_source_id=str(c.relation_source_id) if c.relation_source_id else None,
                relation_target_id=str(c.relation_target_id) if c.relation_target_id else None,
                fact_type=c.fact_type.value,
                confidence=c.confidence.value,
                scope=c.scope,
                sensitivity=c.sensitivity.value,
                routing=c.routing.value,
                provenance=c.provenance,
                conflicts_with=[str(cid) for cid in c.conflicts_with],
                raw_extraction=c.raw_extraction,
            )
            self._session.add(row)
        await self._session.flush()

        # 8. Record extraction metrics (GAP-2: §14.1A)
        try:
            from percos.engine.evaluation import get_evaluation_harness
            evaluation = get_evaluation_harness()
            extracted_entities = [
                {"entity_type": c.entity_type, "name": c.entity_data.get("name", ""),
                 "entity_data": c.entity_data}
                for c in candidates
            ]
            extracted_relations = [
                {"relation_type": c.relation_type or "",
                 "source_type": "", "target_type": ""}
                for c in candidates if c.relation_type
            ]
            evaluation.record_extraction_result(
                event_id=event_id,
                extracted_entities=extracted_entities,
                extracted_relations=extracted_relations or None,
            )
        except Exception:
            log.debug("extraction_metrics_recording_failed")

        # 9. Generate schema-gap proposals for unknown entity types (GAP-3: §8.3)
        try:
            from percos.schema import get_domain_schema
            schema = get_domain_schema()
            known_types = {t.lower() for t in schema.get_entity_type_names()}
            for c in candidates:
                if c.entity_type and c.entity_type.lower() not in known_types:
                    await self._emit_schema_gap_proposal(c)
        except Exception:
            log.debug("schema_gap_proposal_failed")

        log.info("compiled", event_id=event_id, candidates=len(candidates))
        return candidates

    async def _emit_schema_gap_proposal(self, candidate: CandidateFact) -> None:
        """Emit an evolution proposal for a discovered but unknown entity type (§8.3)."""
        if not self._evolution:
            return
        try:
            from percos.models.events import EvolutionProposal
            proposal = EvolutionProposal(
                change_type="ontology_extension",
                description=f"Schema-gap: discovered unknown entity type '{candidate.entity_type}'",
                payload={
                    "entity_type": candidate.entity_type,
                    "sample_data": candidate.entity_data,
                    "source_event": str(candidate.event_id),
                },
            )
            await self._evolution.propose(proposal)
            log.info("schema_gap_proposal_emitted", entity_type=candidate.entity_type)
        except Exception as exc:
            log.debug("schema_gap_proposal_emit_failed", error=str(exc))

    # ── Ontology schema validation (§6.1 – GAP-C6) ───────

    @staticmethod
    def _validate_against_ontology(entity_type: str, entity_data: dict) -> list[str]:
        """Validate entity_data against the corresponding ontology Pydantic model.

        Returns a list of warning strings.  An empty list means the data
        passed validation (or no model is registered for the type).
        """
        model_map = _get_ontology_model_map()
        model_cls = model_map.get(entity_type.lower())
        if model_cls is None:
            return []  # no model registered – skip validation
        try:
            # Construct with name from entity_data (required by Entity base)
            data = {**entity_data}
            if "name" not in data:
                data["name"] = data.get("description", entity_type)
            model_cls(**data)
            return []
        except Exception as exc:
            return [str(exc)]

    async def _detect_conflicts(self, candidate: CandidateFact) -> list[UUID]:
        """Check for existing facts that conflict with this candidate."""
        conflicts: list[UUID] = []
        existing = await self._semantic.get_active_facts(candidate.entity_type)
        candidate_name = candidate.entity_data.get("name", "").lower()
        if not candidate_name:
            return conflicts

        for fact_row in existing:
            existing_name = fact_row.entity_data.get("name", "").lower()
            if existing_name and existing_name == candidate_name:
                # Same name, same type → potential conflict / update
                conflicts.append(UUID(fact_row.id))
        return conflicts

    def _route(self, candidate: CandidateFact) -> CandidateRouting:
        """Determine routing for a candidate fact."""
        # Secret sensitivity → always quarantine first (before any auto-accept)
        if candidate.sensitivity == Sensitivity.SECRET:
            return CandidateRouting.QUARANTINE

        # High-confidence observed facts with no conflicts → auto-accept
        # Must also meet minimum confidence threshold (Gap #13)
        if (
            candidate.fact_type == FactType.OBSERVED
            and candidate.confidence == Confidence.HIGH
            and not candidate.conflicts_with
            and meets_confidence_threshold(candidate.confidence)
        ):
            return CandidateRouting.AUTO_ACCEPT

        # Low confidence → always needs verification regardless
        if not meets_confidence_threshold(candidate.confidence):
            return CandidateRouting.NEEDS_VERIFICATION

        # Hypotheses always need verification
        if candidate.fact_type == FactType.HYPOTHESIS:
            return CandidateRouting.NEEDS_VERIFICATION

        # Policy facts need user confirmation
        if candidate.fact_type == FactType.POLICY:
            return CandidateRouting.NEEDS_USER_CONFIRM

        # Conflicts → user confirmation
        if candidate.conflicts_with:
            return CandidateRouting.NEEDS_USER_CONFIRM

        return CandidateRouting.NEEDS_VERIFICATION

    async def validate_and_commit(self, candidate_id: str, accept: bool = True) -> str | None:
        """Validate a candidate fact and commit it to semantic memory.

        Core interface: validate_candidate(candidate) -> {accept|confirm|quarantine}

        Also commits extracted relations as RelationRow entries (§7.4 step 3,
        GAP-C2) and returns a tuple-style result via the ``committed_relations``
        attribute on the returned fact_id (kept as plain str for API compat).
        """
        stmt = select(CandidateFactRow).where(CandidateFactRow.id == candidate_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if not row:
            return None

        if not accept:
            row.routing = CandidateRouting.QUARANTINE.value
            await self._session.flush()
            return None

        fact = CommittedFact(
            candidate_id=UUID(row.id),
            entity_type=row.entity_type,
            entity_data=row.entity_data,
            fact_type=FactType(row.fact_type),
            confidence=Confidence(row.confidence),
            scope=row.scope,
            sensitivity=Sensitivity(row.sensitivity),
            source=f"event:{row.event_id}",
            provenance_chain=row.provenance,
        )

        # If there are conflicts, supersede the old facts
        for conflict_id in row.conflicts_with:
            await self._semantic.supersede(conflict_id, fact)
            fact_id = str(fact.id)
            # Commit relation after supersession
            await self._commit_relation_if_present(row, fact_id)
            return fact_id

        fact_id = await self._semantic.commit(fact)
        row.routing = "committed"
        await self._session.flush()

        # Commit relation if the candidate carries relation data (§7.4 – GAP-C2)
        await self._commit_relation_if_present(row, fact_id)

        log.info("fact_committed", fact_id=fact_id, entity_type=row.entity_type)
        return fact_id

    async def _commit_relation_if_present(
        self, row: CandidateFactRow, fact_id: str,
    ) -> str | None:
        """If the candidate row carries relation data, resolve endpoints and create a RelationRow.

        Returns the relation_id if created, else None.
        """
        if not row.relation_type:
            return None

        source_id = row.relation_source_id
        target_id = row.relation_target_id

        # If source/target were not resolved during compile, try to resolve now
        # using the entity name stored in raw_extraction.
        if not source_id or not target_id:
            try:
                raw = json.loads(row.raw_extraction) if row.raw_extraction else {}
            except (json.JSONDecodeError, TypeError):
                raw = {}
            if not source_id and raw.get("relation_source"):
                source_id = await self._resolve_entity_by_name(raw["relation_source"])
            if not target_id and raw.get("relation_target"):
                target_id = await self._resolve_entity_by_name(raw["relation_target"])

        # Fall back: use the newly committed fact_id as source when its entity
        # name matches the relation_source from extraction.
        if not source_id:
            source_id = fact_id
        if not target_id:
            log.warning(
                "relation_target_unresolved",
                candidate_id=row.id,
                relation_type=row.relation_type,
            )
            return None

        relation_id = await self._semantic.add_relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=row.relation_type,
            provenance=[f"candidate:{row.id}", f"event:{row.event_id}"],
        )
        log.info(
            "relation_committed",
            relation_id=relation_id,
            relation_type=row.relation_type,
            source_id=source_id,
            target_id=target_id,
        )
        return relation_id

    async def auto_commit_accepted(self, candidates: list[CandidateFact]) -> list[str]:
        """Auto-commit candidates routed as AUTO_ACCEPT."""
        committed_ids: list[str] = []
        for c in candidates:
            if c.routing == CandidateRouting.AUTO_ACCEPT:
                fid = await self.validate_and_commit(str(c.id), accept=True)
                if fid:
                    committed_ids.append(fid)
        return committed_ids

    # ── Entity resolution helpers ────────────────────────

    async def _link_to_existing_entity(self, candidate: CandidateFact) -> None:
        """GAP-M4: Link extracted entity to an existing KG node if one matches.

        If an active fact with the same entity_type and name already exists,
        annotate the candidate with ``_linked_fact_id`` so downstream logic
        can merge/update rather than create a duplicate.  When the existing
        entity has richer data, supplementary fields from the candidate are
        merged into the candidate's ``entity_data`` under ``_merge_fields``.
        """
        entity_name = (candidate.entity_data or {}).get("name", "")
        if not entity_name:
            return

        existing_id = await self._resolve_entity_by_name(entity_name)
        if not existing_id:
            return

        # Fetch the existing fact to compare
        existing_facts = await self._semantic.get_active_facts(candidate.entity_type)
        existing_row = None
        for f in existing_facts:
            if f.id == existing_id:
                existing_row = f
                break

        if existing_row is None:
            # Name matched a different entity type — still record the link
            candidate.entity_data["_linked_fact_id"] = existing_id
            candidate.provenance.append(f"linked:{existing_id}")
            log.info("entity_linked_cross_type",
                     name=entity_name, linked_fact_id=existing_id,
                     candidate_type=candidate.entity_type)
            return

        # Same type — mark as linked and compute any new fields the candidate adds
        candidate.entity_data["_linked_fact_id"] = existing_id
        candidate.provenance.append(f"linked:{existing_id}")

        existing_data = existing_row.entity_data or {}
        new_fields = {
            k: v for k, v in (candidate.entity_data or {}).items()
            if k not in ("_linked_fact_id", "_ontology_warnings", "name")
            and k not in existing_data
            and v  # skip empty values
        }
        if new_fields:
            candidate.entity_data["_merge_fields"] = new_fields
            log.info("entity_linked_with_merge",
                     name=entity_name, linked_fact_id=existing_id,
                     merge_fields=list(new_fields.keys()))
        else:
            log.info("entity_linked",
                     name=entity_name, linked_fact_id=existing_id)

    async def _resolve_entity_by_name(self, name: str) -> str | None:
        """Resolve an entity name to an existing committed fact ID.

        Performs a case-insensitive match on the ``name`` field inside
        ``entity_data`` across all active committed facts.
        """
        if not name:
            return None
        name_lower = name.strip().lower()
        active_facts = await self._semantic.get_active_facts()
        for f in active_facts:
            existing_name = (f.entity_data or {}).get("name", "")
            if existing_name and existing_name.strip().lower() == name_lower:
                return f.id
        return None

    async def get_committed_relations(self, candidate_id: str) -> list[dict]:
        """Return relations committed for a given candidate (for KG sync)."""
        from sqlalchemy import cast, String
        from percos.stores.tables import RelationRow as _RR

        # Use LIKE on the JSON text representation for SQLite compatibility
        search_token = f"candidate:{candidate_id}"
        stmt = select(_RR).where(
            cast(_RR.provenance_chain, String).like(f"%{search_token}%")
        )
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        return [
            {
                "relation_id": r.id,
                "source_id": r.source_id,
                "target_id": r.target_id,
                "relation_type": r.relation_type,
            }
            for r in rows
        ]
