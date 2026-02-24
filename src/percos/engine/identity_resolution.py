"""Multi-source identity resolution and entity deduplication (GAP-L5).

Provides:
- Fuzzy name matching across committed facts (Levenshtein / token overlap).
- Duplicate-pair detection with a configurable similarity threshold.
- Entity merging: retargets all references from duplicate IDs to a canonical ID.
- Alias tracking in ``entity_data.aliases`` to preserve original names.
"""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from percos.logging import get_logger
from percos.stores.tables import CommittedFactRow, RelationRow

log = get_logger("identity_resolution")

# ── Similarity helpers ──────────────────────────────────

def _normalize(name: str) -> str:
    """Lower-case, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity over word tokens."""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _sequence_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio (Levenshtein-like)."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def entity_similarity(a: str, b: str) -> float:
    """Combined similarity score (max of token overlap and sequence ratio)."""
    return max(_token_overlap(a, b), _sequence_similarity(a, b))


# ── Identity Resolver ──────────────────────────────────

class IdentityResolver:
    """Detect and merge duplicate entities across committed facts.

    Usage:
        resolver = IdentityResolver(session, threshold=0.85)
        dupes = await resolver.find_duplicates()
        result = await resolver.merge("canonical-id", ["dup-id-1", "dup-id-2"])
    """

    def __init__(
        self,
        session: AsyncSession,
        threshold: float = 0.85,
    ):
        self._session = session
        self._threshold = threshold

    async def find_duplicates(
        self,
        entity_type: str | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Scan committed facts for potential duplicate entities.

        Returns a list of duplicate-pair dicts:
        ``[{"fact_a": {...}, "fact_b": {...}, "similarity": 0.92, "field": "name"}, ...]``
        """
        thresh = threshold or self._threshold
        stmt = select(CommittedFactRow)
        if entity_type:
            stmt = stmt.where(CommittedFactRow.entity_type == entity_type)
        rows = list((await self._session.execute(stmt)).scalars().all())

        # Extract "name" or primary label from entity_data
        labelled: list[tuple[CommittedFactRow, str]] = []
        for r in rows:
            data = r.entity_data or {}
            name = (
                data.get("name")
                or data.get("title")
                or data.get("label")
                or data.get("entity", "")
            )
            if name:
                labelled.append((r, str(name)))

        # Pair-wise comparison (O(n²) – fine for moderate entity counts)
        duplicates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for i, (ra, na) in enumerate(labelled):
            for j in range(i + 1, len(labelled)):
                rb, nb = labelled[j]
                pair_key = "|".join(sorted([ra.id, rb.id]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                sim = entity_similarity(na, nb)
                if sim >= thresh:
                    duplicates.append({
                        "fact_a": {
                            "id": ra.id,
                            "entity_type": ra.entity_type,
                            "name": na,
                            "entity_data": ra.entity_data,
                        },
                        "fact_b": {
                            "id": rb.id,
                            "entity_type": rb.entity_type,
                            "name": nb,
                            "entity_data": rb.entity_data,
                        },
                        "similarity": round(sim, 3),
                        "field": "name",
                    })

        log.info("duplicates_scanned", total_facts=len(labelled), duplicates_found=len(duplicates))
        return duplicates

    async def merge(
        self,
        canonical_id: str,
        duplicate_ids: list[str],
    ) -> dict[str, Any]:
        """Merge duplicate entities into a canonical fact.

        - Collects aliases from duplicates into ``canonical.entity_data.aliases``.
        - Retargets all ``RelationRow`` references to the canonical ID.
        - Deletes the duplicate ``CommittedFactRow`` entries.

        Returns:
            ``{"merged": <count>, "canonical_id": <id>}``
        """
        canonical = (await self._session.execute(
            select(CommittedFactRow).where(CommittedFactRow.id == canonical_id)
        )).scalar_one_or_none()

        if canonical is None:
            return {"merged": 0, "canonical_id": canonical_id, "error": "Canonical fact not found"}

        aliases: list[str] = list(canonical.entity_data.get("aliases", []))
        merged_count = 0

        for dup_id in duplicate_ids:
            if dup_id == canonical_id:
                continue

            dup = (await self._session.execute(
                select(CommittedFactRow).where(CommittedFactRow.id == dup_id)
            )).scalar_one_or_none()

            if dup is None:
                continue

            # Collect alias
            dup_name = (
                dup.entity_data.get("name")
                or dup.entity_data.get("title")
                or dup.entity_data.get("label")
                or ""
            )
            if dup_name and dup_name not in aliases:
                aliases.append(dup_name)
            # Also collect any existing aliases from the duplicate
            for a in dup.entity_data.get("aliases", []):
                if a not in aliases:
                    aliases.append(a)

            # Merge supplementary data fields (don't overwrite canonical)
            for key, value in dup.entity_data.items():
                if key not in ("name", "title", "label", "aliases"):
                    if key not in canonical.entity_data:
                        canonical.entity_data[key] = value

            # Retarget relations: source
            await self._session.execute(
                update(RelationRow)
                .where(RelationRow.source_id == dup_id)
                .values(source_id=canonical_id)
            )
            # Retarget relations: target
            await self._session.execute(
                update(RelationRow)
                .where(RelationRow.target_id == dup_id)
                .values(target_id=canonical_id)
            )

            # Delete duplicate
            await self._session.delete(dup)
            merged_count += 1

        # Update canonical with collected aliases
        canonical.entity_data = {**canonical.entity_data, "aliases": aliases}
        await self._session.flush()

        log.info("entities_merged", canonical=canonical_id, merged=merged_count)
        return {"merged": merged_count, "canonical_id": canonical_id}

    async def suggest_canonical(
        self,
        fact_ids: list[str],
    ) -> str | None:
        """Suggest which fact should be the canonical one.

        Heuristic: prefer the fact with the most data fields, highest
        confidence, and earliest creation date.
        """
        from percos.engine.security import CONFIDENCE_ORDER

        best_id: str | None = None
        best_score = -1

        for fid in fact_ids:
            row = (await self._session.execute(
                select(CommittedFactRow).where(CommittedFactRow.id == fid)
            )).scalar_one_or_none()
            if row is None:
                continue

            data_fields = len(row.entity_data) if row.entity_data else 0
            conf = CONFIDENCE_ORDER.get(row.confidence, 0)
            score = data_fields * 10 + conf * 5
            if score > best_score:
                best_score = score
                best_id = fid

        return best_id
