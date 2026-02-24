"""Semantic memory store – committed facts with temporal metadata + knowledge graph."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from percos.models.enums import BeliefStatus
from percos.models.events import CommittedFact
from percos.stores.tables import CommittedFactRow, RelationRow


class SemanticStore:
    """Manages committed facts (what-is-true/believed) in the semantic memory."""

    def __init__(self, session: AsyncSession, chroma_collection=None):
        self._session = session
        self._chroma = chroma_collection

    # ── Commit a fact ───────────────────────────────────
    async def commit(self, fact: CommittedFact) -> str:
        from datetime import datetime, timezone as _tz
        now = datetime.now(tz=_tz.utc)
        row = CommittedFactRow(
            id=str(fact.id),
            candidate_id=str(fact.candidate_id),
            entity_type=fact.entity_type,
            entity_data=fact.entity_data,
            fact_type=fact.fact_type.value,
            confidence=fact.confidence.value,
            scope=fact.scope,
            sensitivity=fact.sensitivity.value,
            source=fact.source,
            created_at=fact.created_at,
            last_verified=fact.last_verified or now,
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            belief_status=BeliefStatus.ACTIVE,
            provenance_chain=fact.provenance_chain,
        )
        self._session.add(row)
        await self._session.flush()

        # Index in ChromaDB for vector search (Gap #5)
        if self._chroma is not None:
            doc_text = self._fact_to_document(fact.entity_type, fact.entity_data)
            if doc_text:
                self._chroma.upsert(
                    ids=[str(fact.id)],
                    documents=[doc_text],
                    metadatas=[{
                        "entity_type": fact.entity_type,
                        "confidence": fact.confidence.value,
                        "scope": fact.scope,
                        "fact_type": fact.fact_type.value,
                    }],
                )

        return str(fact.id)

    @staticmethod
    def _fact_to_document(entity_type: str, entity_data: dict) -> str:
        """Convert a fact to a searchable text document."""
        parts = [f"[{entity_type}]"]
        name = entity_data.get("name", "")
        if name:
            parts.append(name)
        desc = entity_data.get("description", "")
        if desc:
            parts.append(desc)
        # Include remaining keys
        for k, v in entity_data.items():
            if k not in ("name", "description") and v:
                parts.append(f"{k}: {v}")
        return " ".join(parts)

    # ── Read ────────────────────────────────────────────
    async def get(self, fact_id: str) -> CommittedFactRow | None:
        stmt = select(CommittedFactRow).where(CommittedFactRow.id == fact_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_facts(self, entity_type: str | None = None) -> list[CommittedFactRow]:
        """Return currently active facts, optionally filtered by entity type."""
        conditions = [CommittedFactRow.belief_status == BeliefStatus.ACTIVE]
        if entity_type:
            conditions.append(CommittedFactRow.entity_type == entity_type)
        stmt = select(CommittedFactRow).where(and_(*conditions)).order_by(CommittedFactRow.created_at.desc())
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_facts_by_scope(self, scope: str) -> list[CommittedFactRow]:
        stmt = (
            select(CommittedFactRow)
            .where(and_(CommittedFactRow.scope == scope, CommittedFactRow.belief_status == BeliefStatus.ACTIVE))
            .order_by(CommittedFactRow.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ── Temporal updates ────────────────────────────────
    async def supersede(self, old_fact_id: str, new_fact: CommittedFact) -> str:
        """Mark old fact as superseded and commit new one."""
        old_row = await self.get(old_fact_id)
        if old_row:
            old_row.belief_status = BeliefStatus.SUPERSEDED
            old_row.valid_to = datetime.now(tz=timezone.utc)
        new_fact.provenance_chain.append(f"supersedes:{old_fact_id}")
        return await self.commit(new_fact)

    async def mark_stale(self, fact_id: str) -> None:
        row = await self.get(fact_id)
        if row:
            row.belief_status = BeliefStatus.STALE

    async def retract(self, fact_id: str) -> None:
        row = await self.get(fact_id)
        if row:
            row.belief_status = BeliefStatus.RETRACTED
            row.valid_to = datetime.now(tz=timezone.utc)

    # ── Provenance / explainability ─────────────────────
    async def explain(self, fact_id: str) -> dict:
        """Return the provenance chain for a fact."""
        row = await self.get(fact_id)
        if not row:
            return {"error": "fact_not_found"}
        return {
            "fact_id": row.id,
            "entity_type": row.entity_type,
            "fact_type": row.fact_type,
            "confidence": row.confidence,
            "source": row.source,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "valid_from": row.valid_from.isoformat() if row.valid_from else None,
            "valid_to": row.valid_to.isoformat() if row.valid_to else None,
            "belief_status": row.belief_status,
            "provenance_chain": row.provenance_chain,
        }

    # ── Relations ───────────────────────────────────────
    async def add_relation(
        self, source_id: str, target_id: str, relation_type: str,
        weight: float = 1.0, provenance: list[str] | None = None,
    ) -> str:
        import uuid
        row = RelationRow(
            id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            provenance_chain=provenance or [],
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def get_relations(self, entity_id: str) -> list[RelationRow]:
        """Get all relations involving an entity (as source or target)."""
        stmt = select(RelationRow).where(
            (RelationRow.source_id == entity_id) | (RelationRow.target_id == entity_id)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def search_facts(self, query: str, n_results: int = 20) -> list[CommittedFactRow]:
        """Semantic search across committed facts.

        Uses ChromaDB vector search when available, falls back to text search.
        """
        if self._chroma is not None:
            return await self._vector_search(query, n_results)
        return await self._text_search(query)

    async def _vector_search(self, query: str, n_results: int = 20) -> list[CommittedFactRow]:
        """Search using ChromaDB semantic similarity."""
        try:
            results = self._chroma.query(query_texts=[query], n_results=n_results)
            fact_ids = results.get("ids", [[]])[0]
            if not fact_ids:
                return []
            # Load the actual CommittedFactRow objects
            stmt = (
                select(CommittedFactRow)
                .where(
                    CommittedFactRow.id.in_(fact_ids),
                    CommittedFactRow.belief_status == BeliefStatus.ACTIVE,
                )
            )
            result = await self._session.execute(stmt)
            rows = list(result.scalars().all())
            # Preserve ChromaDB ranking order
            id_order = {fid: idx for idx, fid in enumerate(fact_ids)}
            rows.sort(key=lambda r: id_order.get(r.id, 999))
            return rows
        except Exception:
            return await self._text_search(query)

    async def _text_search(self, query: str) -> list[CommittedFactRow]:
        """Fallback: client-side text search."""
        stmt = (
            select(CommittedFactRow)
            .where(CommittedFactRow.belief_status == BeliefStatus.ACTIVE)
            .order_by(CommittedFactRow.created_at.desc())
            .limit(50)
        )
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        query_lower = query.lower()
        return [r for r in rows if query_lower in str(r.entity_data).lower()]
