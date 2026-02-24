"""Episodic memory store – append-only relational rows + vector index via ChromaDB."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from percos.models.events import EpisodicEntry
from percos.stores.tables import EpisodicRow


class EpisodicStore:
    """Manages episodic (what-happened) memory in SQL + ChromaDB."""

    def __init__(self, session: AsyncSession, chroma_collection=None):
        self._session = session
        self._chroma = chroma_collection   # optional ChromaDB collection

    # ── Write ───────────────────────────────────────────
    async def append(self, entry: EpisodicEntry) -> str:
        """Append an episodic entry (append-only)."""
        row = EpisodicRow(
            id=str(entry.id),
            event_id=str(entry.event_id),
            memory_type=entry.memory_type.value,
            timestamp=entry.timestamp,
            content=entry.content,
            metadata_extra=entry.metadata_extra,
        )
        self._session.add(row)
        await self._session.flush()

        # Index in vector store if collection provided
        if self._chroma is not None and entry.content:
            self._chroma.upsert(
                ids=[str(entry.id)],
                documents=[entry.content],
                metadatas=[{
                    "event_id": str(entry.event_id),
                    "timestamp": entry.timestamp.isoformat(),
                }],
            )
        return str(entry.id)

    # ── Read ────────────────────────────────────────────
    async def get(self, entry_id: str) -> EpisodicRow | None:
        stmt = select(EpisodicRow).where(EpisodicRow.id == entry_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_event(self, event_id: str) -> list[EpisodicRow]:
        stmt = select(EpisodicRow).where(EpisodicRow.event_id == event_id).order_by(EpisodicRow.timestamp)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_recent(self, limit: int = 20) -> list[EpisodicRow]:
        stmt = select(EpisodicRow).order_by(EpisodicRow.timestamp.desc()).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ── Vector search ───────────────────────────────────
    def search_similar(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Semantic similarity search via ChromaDB."""
        if self._chroma is None:
            return []
        results = self._chroma.query(query_texts=[query_text], n_results=n_results)
        items = []
        for i, doc_id in enumerate(results["ids"][0]):
            items.append({
                "id": doc_id,
                "document": results["documents"][0][i] if results["documents"] else "",
                "distance": results["distances"][0][i] if results.get("distances") else None,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })
        return items
