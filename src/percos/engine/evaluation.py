"""Evaluation harness (§13) – metrics framework for the cognitive pipeline.

Tracks:
- Retrieval quality (precision, recall, relevance)
- Memory accuracy (fact consistency, staleness rate)
- Response quality (coherence, helpfulness)
- Pipeline latency
- Evolution effectiveness
- Agent quality (plan success, policy compliance, unsafe actions, abstentions, corrections)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from percos.logging import get_logger

log = get_logger("evaluation")


@dataclass
class MetricSample:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationHarness:
    """Collects and reports metrics for the cognitive OS pipeline."""

    def __init__(self) -> None:
        self._metrics: dict[str, list[MetricSample]] = {}
        self._timers: dict[str, float] = {}
        # Ground-truth store for precision/recall evaluation
        self._ground_truth: dict[str, set[str]] = {}
        # Agent outcome tracking
        self._plan_outcomes: list[bool] = []
        self._policy_checks: list[bool] = []
        self._unsafe_actions: int = 0
        self._abstentions: int = 0
        self._user_corrections: int = 0
        self._total_actions: int = 0

    # ── Recording ───────────────────────────────────────

    def record(self, name: str, value: float, **metadata: Any) -> None:
        """Record a metric sample."""
        sample = MetricSample(name=name, value=value, metadata=metadata)
        self._metrics.setdefault(name, []).append(sample)

    def start_timer(self, name: str) -> None:
        """Start a latency timer."""
        self._timers[name] = time.monotonic()

    def stop_timer(self, name: str) -> float:
        """Stop timer and record latency in ms."""
        start = self._timers.pop(name, None)
        if start is None:
            return 0.0
        elapsed_ms = (time.monotonic() - start) * 1000
        self.record(f"latency.{name}", elapsed_ms)
        return elapsed_ms

    # ── Retrieval metrics (§13.1A) ──────────────────────

    def set_ground_truth(self, query: str, relevant_fact_ids: set[str]) -> None:
        """Register ground-truth relevant fact IDs for a query (for precision/recall)."""
        self._ground_truth[query] = relevant_fact_ids

    def record_retrieval(
        self,
        query: str,
        semantic_count: int,
        episodic_count: int,
        graph_entities: int,
        relevance_score: float | None = None,
        retrieved_fact_ids: list[str] | None = None,
    ) -> None:
        """Record retrieval quality metrics.

        If ground truth was registered via ``set_ground_truth`` and
        ``retrieved_fact_ids`` is provided, precision and recall are
        computed automatically.
        """
        self.record("retrieval.semantic_count", float(semantic_count), query=query)
        self.record("retrieval.episodic_count", float(episodic_count), query=query)
        self.record("retrieval.graph_entities", float(graph_entities), query=query)
        if relevance_score is not None:
            self.record("retrieval.relevance", relevance_score, query=query)

        # Precision / recall against ground truth
        if retrieved_fact_ids is not None and query in self._ground_truth:
            gt = self._ground_truth[query]
            retrieved_set = set(retrieved_fact_ids)
            tp = len(retrieved_set & gt)
            precision = tp / len(retrieved_set) if retrieved_set else 0.0
            recall = tp / len(gt) if gt else 0.0
            self.record("retrieval.precision", precision, query=query)
            self.record("retrieval.recall", recall, query=query)

    # ── Memory metrics (§13.1A) ─────────────────────────

    def record_memory_stats(
        self,
        active_facts: int,
        stale_facts: int,
        contradictions: int,
    ) -> None:
        """Record memory health metrics."""
        total = active_facts + stale_facts
        staleness_rate = stale_facts / total if total > 0 else 0.0
        self.record("memory.active_facts", float(active_facts))
        self.record("memory.stale_facts", float(stale_facts))
        self.record("memory.staleness_rate", staleness_rate)
        self.record("memory.contradictions", float(contradictions))

    def record_belief_accuracy(
        self,
        correct: int,
        incorrect: int,
    ) -> None:
        """Record belief accuracy after a user correction or verification pass.

        §13.1A — Belief accuracy.
        """
        total = correct + incorrect
        accuracy = correct / total if total > 0 else 0.0
        self.record("memory.belief_accuracy", accuracy,
                     correct=correct, incorrect=incorrect)

    def record_update_correctness(self, correct: bool) -> None:
        """Record whether a single memory update was later judged correct.

        §13.1A — Update correctness.
        """
        self.record("memory.update_correctness", 1.0 if correct else 0.0)

    def record_provenance_completeness(
        self,
        fact_id: str,
        chain_length: int,
        has_source: bool,
        has_timestamps: bool,
    ) -> None:
        """Score provenance completeness for a fact (0-1).

        §13.1A — Provenance completeness.
        """
        score = 0.0
        if chain_length > 0:
            score += 0.4
        if has_source:
            score += 0.3
        if has_timestamps:
            score += 0.3
        self.record("memory.provenance_completeness", score, fact_id=fact_id)

    def record_temporal_consistency(
        self,
        valid_facts: int,
        overlapping_contradictions: int,
    ) -> None:
        """Track temporal consistency — facts that contradict within the same time window.

        §13.1A — Temporal consistency.
        """
        total = valid_facts + overlapping_contradictions
        consistency = valid_facts / total if total > 0 else 1.0
        self.record("memory.temporal_consistency", consistency)

    # ── Agent quality metrics (§13.1B) ──────────────────

    def record_plan_outcome(self, success: bool) -> None:
        """Record whether a plan succeeded. §13.1B — Plan success rate."""
        self._plan_outcomes.append(success)
        self.record("agent.plan_outcome", 1.0 if success else 0.0)

    def record_policy_compliance(self, compliant: bool) -> None:
        """Record whether an action was policy-compliant. §13.1B — Policy compliance."""
        self._policy_checks.append(compliant)
        self.record("agent.policy_compliance", 1.0 if compliant else 0.0)

    def record_unsafe_action(self) -> None:
        """Record an unsafe action. §13.1B — Unsafe action rate."""
        self._unsafe_actions += 1
        self._total_actions += 1
        self.record("agent.unsafe_action", 1.0)

    def record_safe_action(self) -> None:
        """Record a safe action (for computing unsafe rate)."""
        self._total_actions += 1

    def record_abstention(self) -> None:
        """Record when the agent abstained due to uncertainty. §13.1B."""
        self._abstentions += 1
        self.record("agent.abstention", 1.0)

    def record_user_correction(self) -> None:
        """Record when the user corrected the agent. §13.1B — User correction frequency."""
        self._user_corrections += 1
        self.record("agent.user_correction", 1.0)

    # ── Compilation metrics ─────────────────────────────

    def record_compilation(
        self,
        candidates_extracted: int,
        auto_committed: int,
        needs_review: int,
    ) -> None:
        """Record memory compilation metrics."""
        self.record("compilation.candidates_extracted", float(candidates_extracted))
        self.record("compilation.auto_committed", float(auto_committed))
        self.record("compilation.needs_review", float(needs_review))
        acceptance_rate = auto_committed / candidates_extracted if candidates_extracted > 0 else 0.0
        self.record("compilation.acceptance_rate", acceptance_rate)

    # ── Extraction-specific metrics ─────────────────────

    def set_extraction_ground_truth(
        self,
        event_id: str,
        expected_entities: list[dict[str, Any]],
        expected_relations: list[dict[str, Any]] | None = None,
    ) -> None:
        """Register expected extraction results for an input event.

        *expected_entities*: list of dicts with at least ``entity_type`` and
        identifying fields (e.g. ``name``).
        *expected_relations*: list of dicts with ``relation_type``,
        ``source_type``, ``target_type``.
        """
        key = f"extraction:{event_id}"
        self._ground_truth[key] = {  # type: ignore[assignment]
            "entities": expected_entities,
            "relations": expected_relations or [],
        }

    def record_extraction_result(
        self,
        event_id: str,
        extracted_entities: list[dict[str, Any]],
        extracted_relations: list[dict[str, Any]] | None = None,
        schema_entity_types: list[str] | None = None,
    ) -> None:
        """Evaluate extracted entities/relations against ground truth.

        Metrics recorded:
        - ``extraction.entity_precision`` / ``extraction.entity_recall``
        - ``extraction.relation_precision`` / ``extraction.relation_recall``
        - ``extraction.schema_conformance`` — fraction of extracted types
          that are valid per the domain schema
        """
        extracted_relations = extracted_relations or []
        key = f"extraction:{event_id}"
        gt = self._ground_truth.get(key)  # type: ignore[arg-type]

        if gt and isinstance(gt, dict):
            # Entity precision / recall (matching on entity_type + name)
            def _entity_key(e: dict) -> str:
                return f"{e.get('entity_type', '')}:{e.get('name', e.get('entity_data', {}).get('name', ''))}".lower()

            gt_entity_keys = {_entity_key(e) for e in gt["entities"]}
            ext_entity_keys = {_entity_key(e) for e in extracted_entities}

            tp = len(gt_entity_keys & ext_entity_keys)
            entity_precision = tp / len(ext_entity_keys) if ext_entity_keys else 0.0
            entity_recall = tp / len(gt_entity_keys) if gt_entity_keys else 0.0
            self.record("extraction.entity_precision", entity_precision, event_id=event_id)
            self.record("extraction.entity_recall", entity_recall, event_id=event_id)

            # Relation precision / recall (matching on type + source_type + target_type)
            def _rel_key(r: dict) -> str:
                return f"{r.get('relation_type', '')}:{r.get('source_type', '')}:{r.get('target_type', '')}".lower()

            gt_rel_keys = {_rel_key(r) for r in gt.get("relations", [])}
            ext_rel_keys = {_rel_key(r) for r in extracted_relations}

            if gt_rel_keys or ext_rel_keys:
                rel_tp = len(gt_rel_keys & ext_rel_keys)
                rel_precision = rel_tp / len(ext_rel_keys) if ext_rel_keys else 0.0
                rel_recall = rel_tp / len(gt_rel_keys) if gt_rel_keys else 0.0
                self.record("extraction.relation_precision", rel_precision, event_id=event_id)
                self.record("extraction.relation_recall", rel_recall, event_id=event_id)

        # Schema conformance: what fraction of extracted entity types are valid?
        if schema_entity_types is not None and extracted_entities:
            valid_types = {t.lower() for t in schema_entity_types}
            conforming = sum(
                1 for e in extracted_entities
                if e.get("entity_type", "").lower() in valid_types
            )
            conformance = conforming / len(extracted_entities)
            self.record("extraction.schema_conformance", conformance, event_id=event_id)
        elif extracted_entities:
            # Auto-load from schema if available
            try:
                from percos.schema import get_domain_schema
                schema = get_domain_schema()
                valid_types = {t.lower() for t in schema.get_entity_type_names()}
                conforming = sum(
                    1 for e in extracted_entities
                    if e.get("entity_type", "").lower() in valid_types
                )
                conformance = conforming / len(extracted_entities)
                self.record("extraction.schema_conformance", conformance, event_id=event_id)
            except Exception:
                pass

    # ── Evolution metrics (§13.1C) ──────────────────────

    def record_evolution(
        self,
        proposals_total: int,
        deployed: int,
        rolled_back: int,
    ) -> None:
        """Record evolution pipeline metrics."""
        self.record("evolution.proposals_total", float(proposals_total))
        self.record("evolution.deployed", float(deployed))
        self.record("evolution.rolled_back", float(rolled_back))
        success_rate = deployed / proposals_total if proposals_total > 0 else 0.0
        self.record("evolution.success_rate", success_rate)

    def record_evolution_improvement(
        self,
        proposal_id: str,
        baseline_score: float,
        new_score: float,
    ) -> None:
        """Record quality improvement after deploying a proposal. §13.1C."""
        delta = new_score - baseline_score
        self.record("evolution.improvement_delta", delta, proposal_id=proposal_id)
        self.record("evolution.post_deploy_score", new_score, proposal_id=proposal_id)

    def record_regression(self, proposal_id: str) -> None:
        """Record a regression detected after a deployment. §13.1C — Regression rate."""
        self.record("evolution.regression", 1.0, proposal_id=proposal_id)

    def record_safety_incident(self, description: str = "") -> None:
        """Record a safety incident after evolution change. §13.1C."""
        self.record("evolution.safety_incident", 1.0, description=description)

    # ── Aggregate computed metrics ──────────────────────

    def get_agent_summary(self) -> dict[str, Any]:
        """Compute aggregate agent quality metrics (§13.1B)."""
        plan_rate = (
            sum(self._plan_outcomes) / len(self._plan_outcomes)
            if self._plan_outcomes else 0.0
        )
        compliance_rate = (
            sum(self._policy_checks) / len(self._policy_checks)
            if self._policy_checks else 0.0
        )
        unsafe_rate = (
            self._unsafe_actions / self._total_actions
            if self._total_actions else 0.0
        )
        return {
            "plan_success_rate": plan_rate,
            "policy_compliance_rate": compliance_rate,
            "unsafe_action_rate": unsafe_rate,
            "abstention_count": self._abstentions,
            "user_correction_count": self._user_corrections,
            "total_actions": self._total_actions,
        }

    # ── Reporting ───────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all collected metrics."""
        summary: dict[str, Any] = {}
        for name, samples in self._metrics.items():
            values = [s.value for s in samples]
            summary[name] = {
                "count": len(values),
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "avg": sum(values) / len(values) if values else 0,
                "latest": values[-1] if values else 0,
            }
        # Merge agent aggregate
        summary["_agent_summary"] = self.get_agent_summary()
        return summary

    def get_latest(self, name: str) -> float | None:
        """Get the latest value for a given metric."""
        samples = self._metrics.get(name, [])
        return samples[-1].value if samples else None

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        self._timers.clear()
        self._ground_truth.clear()
        self._plan_outcomes.clear()
        self._policy_checks.clear()
        self._unsafe_actions = 0
        self._abstentions = 0
        self._user_corrections = 0
        self._total_actions = 0

    def export_all(self) -> list[dict[str, Any]]:
        """Export all metric samples as dicts."""
        result = []
        for name, samples in self._metrics.items():
            for s in samples:
                result.append({
                    "name": s.name,
                    "value": s.value,
                    "timestamp": s.timestamp.isoformat(),
                    "metadata": s.metadata,
                })
        return result


    # ── Persistence (§13 – survive restarts) ────────────

    async def save_to_db(self, session) -> int:
        """Persist all metric samples to the database and clear in-memory buffer.

        Returns the number of rows written.
        """
        from percos.stores.tables import MetricRow

        rows_written = 0
        for name, samples in self._metrics.items():
            for s in samples:
                row = MetricRow(
                    name=s.name,
                    value=s.value,
                    timestamp=s.timestamp,
                    metadata_extra=s.metadata,
                )
                session.add(row)
                rows_written += 1
        if rows_written:
            await session.flush()
        # Keep the aggregate counters but clear sample buffer to avoid duplication
        self._metrics.clear()
        log.info("metrics_persisted", rows=rows_written)
        return rows_written

    async def load_from_db(self, session, since: datetime | None = None) -> None:
        """Reload metric samples from the database.

        Restores aggregate counters from persisted agent-quality samples.
        """
        from sqlalchemy import select
        from percos.stores.tables import MetricRow

        stmt = select(MetricRow).order_by(MetricRow.timestamp)
        if since:
            stmt = stmt.where(MetricRow.timestamp >= since)
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        for r in rows:
            sample = MetricSample(
                name=r.name,
                value=r.value,
                timestamp=r.timestamp,
                metadata=r.metadata_extra or {},
            )
            self._metrics.setdefault(r.name, []).append(sample)
            # Restore aggregate counters
            if r.name == "agent.plan_outcome":
                self._plan_outcomes.append(r.value == 1.0)
            elif r.name == "agent.policy_compliance":
                self._policy_checks.append(r.value == 1.0)
            elif r.name == "agent.unsafe_action":
                self._unsafe_actions += 1
                self._total_actions += 1
            elif r.name == "agent.abstention":
                self._abstentions += 1
            elif r.name == "agent.user_correction":
                self._user_corrections += 1
        # Count safe actions (total actions recorded minus unsafe)
        safe_samples = self._metrics.get("agent.plan_outcome", [])
        if safe_samples and self._total_actions < len(safe_samples):
            self._total_actions = len(safe_samples)
        log.info("metrics_loaded_from_db", samples=len(rows))

    async def query_history(
        self,
        session,
        metric_name: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Query persisted metric history with optional filters (GAP-L6).

        Args:
            session: AsyncSession to use for the query.
            metric_name: Filter by metric name (exact match). None = all metrics.
            since: Only samples on or after this timestamp.
            until: Only samples on or before this timestamp.
            limit: Max number of samples to return.

        Returns:
            List of metric sample dicts with name, value, timestamp, metadata.
        """
        from sqlalchemy import select as sql_select
        from percos.stores.tables import MetricRow

        stmt = sql_select(MetricRow).order_by(MetricRow.timestamp.desc())
        if metric_name:
            stmt = stmt.where(MetricRow.name == metric_name)
        if since:
            stmt = stmt.where(MetricRow.timestamp >= since)
        if until:
            stmt = stmt.where(MetricRow.timestamp <= until)
        stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        rows = list(result.scalars().all())

        return [
            {
                "name": r.name,
                "value": r.value,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "metadata": r.metadata_extra or {},
            }
            for r in rows
        ]


# Module-level singleton
_harness: EvaluationHarness | None = None


def get_evaluation_harness() -> EvaluationHarness:
    """Return the singleton evaluation harness."""
    global _harness
    if _harness is None:
        _harness = EvaluationHarness()
    return _harness
