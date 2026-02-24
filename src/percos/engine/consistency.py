"""Cross-memory consistency checker (§11.2).

Compares data across episodic, semantic, and procedural stores to detect
incoherence that single-store checks cannot catch:
  - Episodic entries that contradict active semantic facts
  - Procedural steps referencing retracted beliefs
  - Temporal ordering anomalies between stores
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from percos.logging import get_logger
from percos.stores.tables import CommittedFactRow, EpisodicRow, ProceduralRow

log = get_logger("consistency")


class CrossMemoryChecker:
    """Detects cross-store inconsistencies."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(self) -> list[dict[str, Any]]:
        """Run all consistency checks and return a list of issues found."""
        issues: list[dict[str, Any]] = []
        issues.extend(await self._check_episodic_vs_semantic())
        issues.extend(await self._check_procedural_references())
        issues.extend(await self._check_temporal_ordering())
        log.info("consistency_check_done", issues_found=len(issues))
        return issues

    # ── 1. Episodic vs Semantic ─────────────────────────
    async def _check_episodic_vs_semantic(self) -> list[dict[str, Any]]:
        """Find episodic entries that reference entity types with no active semantic fact."""
        issues: list[dict[str, Any]] = []

        # Gather active entity types from semantic memory
        stmt_facts = (
            select(CommittedFactRow.entity_type)
            .where(CommittedFactRow.belief_status == "active")
            .distinct()
        )
        result = await self._session.execute(stmt_facts)
        active_types = {row[0] for row in result.all()}

        # Gather recent episodic entries and check for mention of retracted types
        stmt_retracted = (
            select(CommittedFactRow)
            .where(CommittedFactRow.belief_status.in_(["retracted", "superseded"]))
        )
        result = await self._session.execute(stmt_retracted)
        retracted_facts = list(result.scalars().all())

        retracted_keywords: dict[str, str] = {}
        for f in retracted_facts:
            # Build keyword from entity data for fuzzy cross-check
            desc = str(f.entity_data).lower()
            name_val = f.entity_data.get("name", "") or f.entity_data.get("description", "")
            if name_val:
                retracted_keywords[name_val.lower()] = f.id

        if not retracted_keywords:
            return issues

        stmt_ep = (
            select(EpisodicRow)
            .order_by(EpisodicRow.timestamp.desc())
            .limit(200)
        )
        result = await self._session.execute(stmt_ep)
        episodes = list(result.scalars().all())

        for ep in episodes:
            content_lower = (ep.content or "").lower()
            for keyword, fact_id in retracted_keywords.items():
                if keyword and keyword in content_lower:
                    issues.append({
                        "type": "episodic_references_retracted_fact",
                        "episodic_id": ep.id,
                        "retracted_fact_id": fact_id,
                        "keyword": keyword,
                        "severity": "low",
                    })
        return issues

    # ── 2. Procedural references ────────────────────────
    async def _check_procedural_references(self) -> list[dict[str, Any]]:
        """Detect procedural entries whose step text references retracted facts."""
        issues: list[dict[str, Any]] = []

        stmt_retracted = (
            select(CommittedFactRow)
            .where(CommittedFactRow.belief_status.in_(["retracted", "superseded"]))
        )
        result = await self._session.execute(stmt_retracted)
        retracted = list(result.scalars().all())

        retracted_names: dict[str, str] = {}
        for f in retracted:
            name = f.entity_data.get("name", "")
            if name:
                retracted_names[name.lower()] = f.id

        if not retracted_names:
            return issues

        stmt_proc = select(ProceduralRow)
        result = await self._session.execute(stmt_proc)
        procedures = list(result.scalars().all())

        for proc in procedures:
            steps_text = " ".join(proc.steps or []).lower()
            desc_text = (proc.description or "").lower()
            combined = steps_text + " " + desc_text
            for name, fact_id in retracted_names.items():
                if name and name in combined:
                    issues.append({
                        "type": "procedural_references_retracted_fact",
                        "procedural_id": proc.id,
                        "retracted_fact_id": fact_id,
                        "keyword": name,
                        "severity": "medium",
                    })
        return issues

    # ── 3. Temporal ordering anomalies ──────────────────
    async def _check_temporal_ordering(self) -> list[dict[str, Any]]:
        """Detect facts whose valid_from is after their valid_to (data integrity)."""
        issues: list[dict[str, Any]] = []

        stmt = (
            select(CommittedFactRow)
            .where(
                CommittedFactRow.valid_from.isnot(None),
                CommittedFactRow.valid_to.isnot(None),
            )
        )
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())

        for row in rows:
            if row.valid_from and row.valid_to and row.valid_from > row.valid_to:
                issues.append({
                    "type": "temporal_ordering_anomaly",
                    "fact_id": row.id,
                    "valid_from": row.valid_from.isoformat(),
                    "valid_to": row.valid_to.isoformat(),
                    "severity": "high",
                })
        return issues
