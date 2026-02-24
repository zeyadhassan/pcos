"""Tests for Medium Priority Gaps (#5–#10).

Covers:
- Gap 5:  Memory Control Panel UI endpoint
- Gap 6:  task_type in query_world_model
- Gap 7:  Standalone plan/execute/reflect API endpoints
- Gap 8:  Context ranking / relevance scores in ContextBundle
- Gap 9:  Procedural memory versioning (update, history, rollback)
- Gap 10: Centralized audit log
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from uuid import uuid4

from percos.models.events import (
    CommittedFact,
    ContextBundle,
    EpisodicEntry,
    PolicyEntry,
    ProceduralEntry,
    WorkingMemory,
)
from percos.models.enums import (
    Confidence,
    FactType,
    Sensitivity,
)
from percos.stores.procedural_policy_stores import ProceduralStore
from percos.stores.audit_log import AuditLog
from percos.stores.tables import ProceduralVersionRow


# ════════════════════════════════════════════════════════
# Gap 6: task_type accepted by query_world_model
# ════════════════════════════════════════════════════════

class TestQueryTaskType:
    """QueryRequest now accepts task_type; retrieve() respects it."""

    def test_query_request_accepts_task_type(self):
        from percos.api.schemas import QueryRequest
        req = QueryRequest(query="What's on my calendar?", task_type="recall")
        assert req.task_type == "recall"

    def test_query_request_task_type_optional(self):
        from percos.api.schemas import QueryRequest
        req = QueryRequest(query="hello")
        assert req.task_type is None

    def test_context_bundle_has_relevance_scores(self):
        """Gap 8: ContextBundle now exposes relevance_scores."""
        bundle = ContextBundle(
            query="test",
            relevance_scores={"fact-1": 0.85, "fact-2": 0.42},
        )
        assert bundle.relevance_scores["fact-1"] == 0.85


# ════════════════════════════════════════════════════════
# Gap 7: Standalone API endpoint schemas
# ════════════════════════════════════════════════════════

class TestStandaloneEndpointSchemas:
    """Schemas for plan/execute/reflect exist and validate correctly."""

    def test_plan_action_request(self):
        from percos.api.schemas import PlanActionRequest
        req = PlanActionRequest(query="Book a meeting", task_type="planning")
        assert req.query == "Book a meeting"
        assert req.task_type == "planning"

    def test_execute_action_request(self):
        from percos.api.schemas import ExecuteActionRequest
        req = ExecuteActionRequest(plan={"goal": "test", "steps": []}, query="do it")
        assert req.plan["goal"] == "test"

    def test_reflect_request(self):
        from percos.api.schemas import ReflectRequest
        req = ReflectRequest(outcome={"success": True, "result": "done"})
        assert req.outcome["success"] is True

    def test_plan_action_response(self):
        from percos.api.schemas import PlanActionResponse
        resp = PlanActionResponse(plan={"plan_id": "123", "steps": []})
        assert resp.plan["plan_id"] == "123"

    def test_execute_action_response(self):
        from percos.api.schemas import ExecuteActionResponse
        resp = ExecuteActionResponse(outcome={"outcome_id": "abc", "success": True})
        assert resp.outcome["success"] is True

    def test_reflect_response(self):
        from percos.api.schemas import ReflectResponse
        resp = ReflectResponse(reflection={"lessons": ["learned something"]})
        assert resp.reflection["lessons"][0] == "learned something"


# ════════════════════════════════════════════════════════
# Gap 8: Context ranking / relevance scores
# ════════════════════════════════════════════════════════

class TestContextRanking:
    """Relevance scores are computed and propagated to ContextBundle."""

    def test_relevance_scores_default_empty(self):
        bundle = ContextBundle(query="test")
        assert bundle.relevance_scores == {}

    def test_relevance_scores_populated(self):
        bundle = ContextBundle(
            query="test",
            relevance_scores={"id-1": 0.9, "id-2": 0.3},
        )
        assert len(bundle.relevance_scores) == 2
        assert bundle.relevance_scores["id-1"] > bundle.relevance_scores["id-2"]

    def test_query_response_includes_relevance(self):
        from percos.api.schemas import QueryResponse
        resp = QueryResponse(
            query="test",
            semantic_facts=[],
            episodic_entries=[],
            policies=[],
            graph_context={},
            relevance_scores={"abc": 0.75},
        )
        assert resp.relevance_scores["abc"] == 0.75


# ════════════════════════════════════════════════════════
# Gap 9: Procedural memory versioning
# ════════════════════════════════════════════════════════

class TestProceduralVersioning:
    """Procedural store supports version increment, history, and rollback."""

    @pytest_asyncio.fixture
    async def proc_store(self, db_session):
        return ProceduralStore(db_session)

    @pytest.mark.asyncio
    async def test_save_creates_initial_version(self, proc_store, db_session):
        """Saving a procedure should create a v1 snapshot."""
        entry = ProceduralEntry(name="greet", description="Say hello", steps=["wave", "say hi"])
        entry_id = await proc_store.save(entry)

        history = await proc_store.get_version_history(entry_id)
        assert len(history) == 1
        assert history[0]["version"] == 1
        assert history[0]["name"] == "greet"

    @pytest.mark.asyncio
    async def test_update_increments_version(self, proc_store, db_session):
        """Updating a procedure should increment version and save history."""
        entry = ProceduralEntry(name="greet", description="v1", steps=["wave"])
        entry_id = await proc_store.save(entry)

        updated = await proc_store.update(entry_id, {
            "description": "v2 improved",
            "steps": ["wave", "smile", "say hi"],
        })
        assert updated is not None
        assert updated.version == 2
        assert updated.description == "v2 improved"
        assert len(updated.steps) == 3

        history = await proc_store.get_version_history(entry_id)
        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[1]["version"] == 1  # snapshot of v1 taken before update

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self, proc_store, db_session):
        result = await proc_store.update("nonexistent-id", {"name": "x"})
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_updates_track_history(self, proc_store, db_session):
        """Three updates should create version 3 with full history."""
        entry = ProceduralEntry(name="workflow", description="v1", steps=["a"])
        entry_id = await proc_store.save(entry)

        await proc_store.update(entry_id, {"description": "v2", "steps": ["a", "b"]})
        await proc_store.update(entry_id, {"description": "v3", "steps": ["a", "b", "c"]})

        row = await proc_store.get(entry_id)
        assert row.version == 3
        assert row.description == "v3"

        history = await proc_store.get_version_history(entry_id)
        # v1 (save), v1 (pre-update-to-v2), v2 (pre-update-to-v3)
        assert len(history) >= 3

    @pytest.mark.asyncio
    async def test_rollback_to_previous_version(self, proc_store, db_session):
        """Rollback should restore the previous version's state."""
        entry = ProceduralEntry(name="deploy", description="v1", steps=["build", "test"])
        entry_id = await proc_store.save(entry)

        await proc_store.update(entry_id, {"description": "v2-broken", "steps": ["build"]})
        row = await proc_store.get(entry_id)
        assert row.version == 2

        rolled_back = await proc_store.rollback(entry_id, target_version=1)
        assert rolled_back is not None
        assert rolled_back.description == "v1"
        assert rolled_back.steps == ["build", "test"]
        assert rolled_back.version == 3  # rollback creates a new version

    @pytest.mark.asyncio
    async def test_rollback_without_target_goes_to_previous(self, proc_store, db_session):
        """Rollback with no target_version rolls back to previous version."""
        entry = ProceduralEntry(name="flow", description="v1", steps=["x"])
        entry_id = await proc_store.save(entry)

        await proc_store.update(entry_id, {"description": "v2", "steps": ["x", "y"]})
        rolled_back = await proc_store.rollback(entry_id)
        assert rolled_back is not None
        assert rolled_back.description == "v1"

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_returns_none(self, proc_store, db_session):
        result = await proc_store.rollback("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_rollback_to_version_zero_returns_none(self, proc_store, db_session):
        entry = ProceduralEntry(name="x", description="v1", steps=[])
        entry_id = await proc_store.save(entry)
        result = await proc_store.rollback(entry_id, target_version=0)
        assert result is None


# ════════════════════════════════════════════════════════
# Gap 10: Centralized audit log
# ════════════════════════════════════════════════════════

class TestAuditLog:
    """Audit log records and queries system actions."""

    @pytest_asyncio.fixture
    async def audit(self, db_session):
        return AuditLog(db_session)

    @pytest.mark.asyncio
    async def test_record_and_query(self, audit, db_session):
        """Recording an action should be queryable."""
        aid = await audit.record(
            "fact_committed", "compiler",
            actor="user", resource_id="fact-123", resource_type="fact",
            details={"entity_type": "preference"},
        )
        assert aid  # non-empty ID

        entries = await audit.query(action="fact_committed")
        assert len(entries) == 1
        assert entries[0]["action"] == "fact_committed"
        assert entries[0]["resource_id"] == "fact-123"
        assert entries[0]["actor"] == "user"

    @pytest.mark.asyncio
    async def test_query_by_component(self, audit, db_session):
        await audit.record("event_ingested", "ingestion", actor="system")
        await audit.record("fact_committed", "compiler", actor="system")

        entries = await audit.query(component="ingestion")
        assert len(entries) == 1
        assert entries[0]["component"] == "ingestion"

    @pytest.mark.asyncio
    async def test_query_by_resource_id(self, audit, db_session):
        await audit.record("belief_updated", "brain", resource_id="abc-123")
        await audit.record("belief_deleted", "brain", resource_id="def-456")

        entries = await audit.query(resource_id="abc-123")
        assert len(entries) == 1
        assert entries[0]["action"] == "belief_updated"

    @pytest.mark.asyncio
    async def test_count(self, audit, db_session):
        await audit.record("fact_committed", "compiler")
        await audit.record("fact_committed", "compiler")
        await audit.record("belief_deleted", "brain")

        assert await audit.count() == 3
        assert await audit.count("fact_committed") == 2
        assert await audit.count("belief_deleted") == 1

    @pytest.mark.asyncio
    async def test_query_limit(self, audit, db_session):
        for i in range(10):
            await audit.record(f"action_{i}", "test")

        entries = await audit.query(limit=3)
        assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_outcome_recorded(self, audit, db_session):
        await audit.record("event_ingested", "ingestion", outcome="blocked")
        entries = await audit.query()
        assert entries[0]["outcome"] == "blocked"

    @pytest.mark.asyncio
    async def test_details_stored(self, audit, db_session):
        await audit.record(
            "maintenance_run", "ttm",
            details={"stale_detected": 5, "marked_stale": 3},
        )
        entries = await audit.query()
        assert entries[0]["details"]["stale_detected"] == 5


# ════════════════════════════════════════════════════════
# Gap 5: Memory Control Panel UI
# ════════════════════════════════════════════════════════

class TestMemoryControlPanel:
    """Panel endpoint serves an HTML page."""

    def test_panel_route_exists(self):
        from percos.api.panel import panel_router
        routes = [r.path for r in panel_router.routes]
        assert "/panel" in routes

    def test_panel_html_content(self):
        from percos.api.panel import PANEL_HTML
        assert "Memory Control Panel" in PANEL_HTML
        assert "Beliefs" in PANEL_HTML
        assert "Audit Log" in PANEL_HTML
        assert "Export" in PANEL_HTML

    def test_panel_registered_in_app(self):
        from percos.app import app
        paths = [r.path for r in app.routes]
        assert "/panel" in paths
