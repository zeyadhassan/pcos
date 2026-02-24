"""Dependency injection helpers for FastAPI routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import chromadb
from sqlalchemy.ext.asyncio import AsyncSession

from percos.config import get_settings
from percos.engine.brain import Brain
from percos.llm import get_llm
from percos.schema import get_domain_schema, DomainSchema
from percos.stores.database import get_session_factory


_chroma_client = None
_chroma_collection = None


def get_chroma_collection():
    """Get or create the shared ChromaDB collection for semantic + episodic memory."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        settings = get_settings()
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="pcos_memory",
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def get_schema() -> DomainSchema:
    """FastAPI dependency: return the active domain schema."""
    return get_domain_schema()


async def get_brain() -> AsyncGenerator[Brain, None]:
    """FastAPI dependency: yields a Brain instance with an active DB session."""
    factory = get_session_factory()
    async with factory() as session:
        llm = get_llm()
        chroma = get_chroma_collection()
        brain = Brain(session, llm, chroma)
        # Restore persisted working memory
        await brain.load_working_memory()
        yield brain


@asynccontextmanager
async def get_brain_standalone() -> AsyncGenerator[Brain, None]:
    """Context manager for creating a Brain outside of a request (e.g. background tasks).

    GAP-M5: Used by the maintenance scheduler to get a Brain with its own session.
    """
    factory = get_session_factory()
    async with factory() as session:
        llm = get_llm()
        chroma = get_chroma_collection()
        brain = Brain(session, llm, chroma)
        await brain.load_working_memory()
        try:
            yield brain
        finally:
            await session.commit()
