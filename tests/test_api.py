"""Tests for the FastAPI API routes."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from percos.app import create_app
from percos.stores.tables import Base


@pytest_asyncio.fixture
async def app_client():
    """Create a test client with an in-memory database."""
    # Create in-memory engine
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    app = create_app()

    # Override the dependency
    from percos.api.deps import get_brain
    from percos.engine.brain import Brain

    class MockLLM:
        async def chat(self, *a, **kw):
            return '{"candidates": []}'
        async def chat_json(self, *a, **kw):
            return {"candidates": []}
        async def extract_structured(self, *a, **kw):
            return {"candidates": [], "intent": "recall", "entities": [], "keywords": []}

    async def override_brain():
        async with session_factory() as session:
            brain = Brain(session, MockLLM())  # type: ignore
            yield brain

    app.dependency_overrides[get_brain] = override_brain

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await engine.dispose()


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, app_client: AsyncClient):
        resp = await app_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestEventIngestionAPI:
    @pytest.mark.asyncio
    async def test_ingest_event(self, app_client: AsyncClient):
        resp = await app_client.post("/api/v1/events/ingest", json={
            "event_type": "conversation",
            "content": "I like morning meetings",
            "source": "test",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "event_id" in data


class TestBeliefsAPI:
    @pytest.mark.asyncio
    async def test_list_beliefs_empty(self, app_client: AsyncClient):
        resp = await app_client.get("/api/v1/beliefs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["beliefs"] == []


class TestChatAPI:
    @pytest.mark.asyncio
    async def test_chat_basic(self, app_client: AsyncClient):
        resp = await app_client.post("/api/v1/chat", json={
            "message": "Hello, what are my tasks?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "event_id" in data
        assert "response" in data


class TestEvolutionAPI:
    @pytest.mark.asyncio
    async def test_list_proposals_empty(self, app_client: AsyncClient):
        resp = await app_client.get("/api/v1/evolution/proposals")
        assert resp.status_code == 200
        data = resp.json()
        assert data["proposals"] == []
