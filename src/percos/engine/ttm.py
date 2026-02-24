"""Temporal Truth Maintenance Engine (§8).

Manages changing truth over time:
- Contradiction detection
- Belief supersession
- Historical preservation
- Staleness detection
- Context-specific truth branching
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from percos.llm import LLMClient
from percos.logging import get_logger
from percos.models.enums import BeliefStatus
from percos.stores.tables import CommittedFactRow

log = get_logger("ttm")

STALENESS_DAYS = 90  # default fallback; per-type staleness from domain schema takes priority


def _get_staleness_days(entity_type: str | None = None) -> int:
    """Return staleness_days for an entity type from the active domain schema.

    Falls back to the global ``STALENESS_DAYS`` constant.
    """
    if entity_type is None:
        return STALENESS_DAYS
    try:
        from percos.schema import get_domain_schema
        schema = get_domain_schema()
        return schema.get_staleness_days(entity_type)
    except Exception:
        return STALENESS_DAYS

CONTRADICTION_SYSTEM_PROMPT = """\
You are a truth-maintenance reasoner for a personal knowledge system.
Given two facts about the same entity, determine if they contradict each other.

Respond with a JSON object:
{
  "contradicts": true/false,
  "explanation": "brief reason",
  "recommended_action": "supersede_old" | "keep_both" | "needs_user_input"
}
"""


class TemporalTruthMaintenance:
    """Manages the temporal consistency of the semantic memory."""

    def __init__(self, session: AsyncSession, llm: LLMClient):
        self._session = session
        self._llm = llm

    # ── Staleness detection ─────────────────────────────
    async def detect_stale_facts(self, days: int | None = None) -> list[CommittedFactRow]:
        """Find active facts that haven't been verified recently.

        When ``days`` is explicitly provided, all entity types use that cutoff.
        Otherwise, each entity type uses its per-type staleness from the domain
        schema (falling back to ``STALENESS_DAYS``).

        Uses ``last_verified`` (§6.3) so that re-confirmed facts are no
        longer treated as stale.  Falls back to ``created_at`` for rows
        that pre-date the column addition.
        """
        if days is not None:
            # Single-cutoff mode (backward compatible)
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
            stmt = (
                select(CommittedFactRow)
                .where(
                    and_(
                        CommittedFactRow.belief_status == BeliefStatus.ACTIVE,
                        CommittedFactRow.last_verified < cutoff,
                    )
                )
                .order_by(CommittedFactRow.last_verified.asc())
            )
            result = await self._session.execute(stmt)
            rows = list(result.scalars().all())
            log.info("staleness_scan", stale_count=len(rows), cutoff_days=days)
            return rows

        # Per-type staleness mode: query all active facts and filter by type
        stmt = (
            select(CommittedFactRow)
            .where(CommittedFactRow.belief_status == BeliefStatus.ACTIVE)
            .order_by(CommittedFactRow.last_verified.asc())
        )
        result = await self._session.execute(stmt)
        all_active = list(result.scalars().all())

        now = datetime.now(tz=timezone.utc)
        stale: list[CommittedFactRow] = []
        for row in all_active:
            type_days = _get_staleness_days(row.entity_type)
            cutoff = now - timedelta(days=type_days)
            last_check = row.last_verified or row.created_at
            if last_check and last_check < cutoff:
                stale.append(row)

        log.info("staleness_scan_per_type", stale_count=len(stale))
        return stale

    async def mark_stale(self, fact_ids: list[str]) -> int:
        """Mark a list of facts as stale."""
        count = 0
        for fid in fact_ids:
            stmt = select(CommittedFactRow).where(CommittedFactRow.id == fid)
            result = await self._session.execute(stmt)
            row = result.scalar_one_or_none()
            if row and row.belief_status == BeliefStatus.ACTIVE:
                row.belief_status = BeliefStatus.STALE
                count += 1
        await self._session.flush()
        log.info("marked_stale", count=count)
        return count

    # ── Contradiction detection ─────────────────────────
    async def check_contradiction(
        self, fact_a: CommittedFactRow, fact_b: CommittedFactRow
    ) -> dict:
        """Use LLM to determine if two facts contradict."""
        user_content = (
            f"Fact A (created {fact_a.created_at.isoformat()}):\n"
            f"  Type: {fact_a.entity_type}\n"
            f"  Data: {fact_a.entity_data}\n\n"
            f"Fact B (created {fact_b.created_at.isoformat()}):\n"
            f"  Type: {fact_b.entity_type}\n"
            f"  Data: {fact_b.entity_data}"
        )
        try:
            result = await self._llm.extract_structured(
                CONTRADICTION_SYSTEM_PROMPT, user_content
            )
            return result if isinstance(result, dict) else {"contradicts": False}
        except Exception as exc:
            log.error("contradiction_check_failed", error=str(exc))
            return {"contradicts": False, "error": str(exc)}

    async def scan_contradictions(self, entity_type: str | None = None) -> list[dict]:
        """Scan active facts for potential contradictions.

        Scope-aware: facts in different scopes are less likely contradictory
        (e.g., a preference at work vs personal scope can coexist).
        """
        conditions = [CommittedFactRow.belief_status == BeliefStatus.ACTIVE]
        if entity_type:
            conditions.append(CommittedFactRow.entity_type == entity_type)
        stmt = select(CommittedFactRow).where(and_(*conditions))
        result = await self._session.execute(stmt)
        facts = list(result.scalars().all())

        # Group by entity name for pairwise contradiction checks
        by_name: dict[str, list[CommittedFactRow]] = {}
        for f in facts:
            name = f.entity_data.get("name", "")
            if name:
                by_name.setdefault(name.lower(), []).append(f)

        contradictions = []
        for name, group in by_name.items():
            if len(group) < 2:
                continue
            # Check pairwise (limited to avoid explosion)
            for i in range(len(group)):
                for j in range(i + 1, min(i + 3, len(group))):
                    fact_a, fact_b = group[i], group[j]

                    # Scope-aware branching (Gap #17):
                    # Facts in different scopes can coexist without contradiction
                    if fact_a.scope != fact_b.scope:
                        continue  # different scopes → not contradictory

                    check = await self.check_contradiction(fact_a, fact_b)
                    if check.get("contradicts"):
                        contradictions.append({
                            "fact_a_id": fact_a.id,
                            "fact_b_id": fact_b.id,
                            "entity_type": entity_type or fact_a.entity_type,
                            "name": name,
                            "scope": fact_a.scope,
                            **check,
                        })

        log.info("contradiction_scan", found=len(contradictions))
        return contradictions

    # ── Supersession ────────────────────────────────────
    async def supersede(self, old_fact_id: str, new_fact_id: str) -> None:
        """Mark old fact as superseded by new fact and flag dependents (§9.3)."""
        old_stmt = select(CommittedFactRow).where(CommittedFactRow.id == old_fact_id)
        result = await self._session.execute(old_stmt)
        old_row = result.scalar_one_or_none()
        if old_row:
            old_row.belief_status = BeliefStatus.SUPERSEDED
            old_row.valid_to = datetime.now(tz=timezone.utc)

        new_stmt = select(CommittedFactRow).where(CommittedFactRow.id == new_fact_id)
        result = await self._session.execute(new_stmt)
        new_row = result.scalar_one_or_none()
        if new_row:
            chain = new_row.provenance_chain or []
            chain.append(f"supersedes:{old_fact_id}")
            new_row.provenance_chain = chain

        # GAP-7: Cascading re-evaluation — flag dependent relations for review
        await self._flag_dependents_for_review(old_fact_id)

        await self._session.flush()
        log.info("superseded", old=old_fact_id, new=new_fact_id)

    async def _flag_dependents_for_review(self, superseded_fact_id: str) -> None:
        """Flag relations and dependent facts that reference a superseded fact (§9.3).

        When a fact is superseded, any relations pointing to/from it may need
        re-evaluation. This creates audit entries for the dependent entities.
        """
        from percos.stores.tables import RelationRow
        # Find relations where this fact is source or target
        stmt = select(RelationRow).where(
            (RelationRow.source_id == superseded_fact_id) |
            (RelationRow.target_id == superseded_fact_id)
        )
        result = await self._session.execute(stmt)
        dependent_relations = list(result.scalars().all())

        for rel in dependent_relations:
            # Determine the other end of the relation
            other_id = rel.target_id if rel.source_id == superseded_fact_id else rel.source_id
            log.info("dependent_flagged_for_review",
                     relation_id=rel.id,
                     relation_type=rel.relation_type,
                     dependent_fact_id=other_id,
                     reason=f"source fact {superseded_fact_id} was superseded")

    # ── Historical query ────────────────────────────────
    async def get_belief_history(self, entity_name: str) -> list[dict]:
        """Get the full history of beliefs about an entity (including superseded)."""
        stmt = (
            select(CommittedFactRow)
            .order_by(CommittedFactRow.created_at.asc())
        )
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        history = []
        name_lower = entity_name.lower()
        for r in rows:
            if name_lower in str(r.entity_data).lower():
                history.append({
                    "fact_id": r.id,
                    "entity_type": r.entity_type,
                    "entity_data": r.entity_data,
                    "belief_status": r.belief_status,
                    "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                    "valid_to": r.valid_to.isoformat() if r.valid_to else None,
                    "confidence": r.confidence,
                    "provenance_chain": r.provenance_chain,
                })
        return history
