"""Tests for LOW priority gap implementations (Gaps 11–15)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from percos.engine.brain import Brain
from percos.engine.consistency import CrossMemoryChecker
from percos.engine.evolution import EvolutionSandbox, FORBIDDEN_CATEGORIES
from percos.engine.style_tracker import CommunicationStyleTracker, StyleProfile
from percos.models.enums import Confidence, FactType, Sensitivity
from percos.models.events import CommittedFact, EpisodicEntry, ProceduralEntry
from percos.stores.episodic_store import EpisodicStore
from percos.stores.semantic_store import SemanticStore
from percos.stores.procedural_policy_stores import ProceduralStore
from percos.stores.tables import CommittedFactRow, EpisodicRow, ProceduralRow


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

class _FakeLLM:
    """Deterministic LLM stub."""

    async def extract_structured(self, system_prompt: str, user_content: str) -> dict:
        return {
            "entities": [],
            "relations": [],
            "valid": True,
            "issues": [],
            "risk_level": "low",
            "recommendation": "approve",
        }

    async def chat(self, system_prompt: str, user_message: str) -> str:
        return '{"response": "ok", "plan": {}, "reflection": {}}'


async def _make_brain(db_session: AsyncSession) -> Brain:
    return Brain(session=db_session, llm=_FakeLLM())


# ─────────────────────────────────────────────────────────
# Gap 11 — Communication Style Tracker
# ─────────────────────────────────────────────────────────

class TestCommunicationStyleTracker:
    """Tests for §14.3 communication style learning."""

    def test_basic_analysis(self):
        tracker = CommunicationStyleTracker()
        features = tracker.analyse("Hello, how are you doing today?")
        assert features["word_count"] == 6
        assert features["is_question"] is True
        assert features["is_greeting"] is True

    def test_emoji_detection(self):
        tracker = CommunicationStyleTracker()
        features = tracker.analyse("Great job! 🎉🎉")
        assert features["has_emoji"] is True
        profile = tracker.get_profile()
        assert profile["emoji_ratio"] == 1.0

    def test_formality_markers(self):
        tracker = CommunicationStyleTracker()
        # Formal message
        tracker.analyse("Furthermore, I would kindly request your attention please.")
        profile = tracker.get_profile()
        assert profile["formality_score"] > 0.5

    def test_informal_markers(self):
        tracker = CommunicationStyleTracker()
        tracker.analyse("hey lol gonna do this btw")
        profile = tracker.get_profile()
        assert profile["formality_score"] < 0.5

    def test_running_average(self):
        tracker = CommunicationStyleTracker()
        tracker.analyse("Short.")
        tracker.analyse("This is a much longer message with many words in it for testing.")
        profile = tracker.get_profile()
        assert profile["messages_analysed"] == 2
        assert profile["avg_word_count"] > 1

    def test_profile_serialisation_roundtrip(self):
        tracker = CommunicationStyleTracker()
        tracker.analyse("Hey, what's up? 🎉")
        tracker.analyse("Please provide more details. Thank you.")
        original = tracker.get_profile()

        # Restore in a new tracker
        tracker2 = CommunicationStyleTracker()
        tracker2.load_profile(original)
        restored = tracker2.get_profile()

        assert restored["messages_analysed"] == original["messages_analysed"]
        assert restored["emoji_ratio"] == original["emoji_ratio"]

    def test_question_ratio(self):
        tracker = CommunicationStyleTracker()
        tracker.analyse("What time is it?")
        tracker.analyse("Tell me the time.")
        profile = tracker.get_profile()
        assert profile["question_ratio"] == 0.5

    def test_exclamation_ratio(self):
        tracker = CommunicationStyleTracker()
        tracker.analyse("Wow! Amazing!")
        tracker.analyse("ok")
        profile = tracker.get_profile()
        assert profile["exclamation_ratio"] == 0.5

    @pytest.mark.asyncio
    async def test_style_integrated_in_brain(self, db_session):
        """Style tracker is optionally integrated into Brain (GAP-5: conditional on schema).

        When the domain schema doesn't define 'communication_style', the tracker
        is None. It can be manually enabled for testing.
        """
        brain = await _make_brain(db_session)
        # Style tracker is None when domain schema doesn't define communication_style
        # This is correct behavior per GAP-5 (domain-agnostic core).
        # Manually enable for testing:
        from percos.engine.style_tracker import CommunicationStyleTracker
        brain.style_tracker = CommunicationStyleTracker()
        assert brain.style_tracker is not None
        brain.style_tracker.analyse("Hello there!")
        profile = brain.style_tracker.get_profile()
        assert profile["messages_analysed"] == 1


# ─────────────────────────────────────────────────────────
# Gap 12 — Cross-Memory Consistency Checks
# ─────────────────────────────────────────────────────────

class TestCrossMemoryConsistency:
    """Tests for §11.2 cross-store coherence checks."""

    @pytest.mark.asyncio
    async def test_no_issues_on_clean_db(self, db_session):
        checker = CrossMemoryChecker(db_session)
        issues = await checker.run()
        assert issues == []

    @pytest.mark.asyncio
    async def test_temporal_ordering_anomaly(self, db_session):
        """Detect a fact where valid_from > valid_to."""
        now = datetime.now(tz=timezone.utc)
        row = CommittedFactRow(
            id=str(uuid4()),
            candidate_id=str(uuid4()),
            entity_type="observation",
            entity_data={"name": "anomaly_test"},
            fact_type="observed",
            confidence="high",
            scope="global",
            sensitivity="internal",
            source="test",
            created_at=now,
            valid_from=now + timedelta(days=10),  # after valid_to → anomaly
            valid_to=now,
            belief_status="active",
            provenance_chain=[],
        )
        db_session.add(row)
        await db_session.flush()

        checker = CrossMemoryChecker(db_session)
        issues = await checker.run()
        anomalies = [i for i in issues if i["type"] == "temporal_ordering_anomaly"]
        assert len(anomalies) >= 1
        assert anomalies[0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_episodic_references_retracted(self, db_session):
        """Detect episodic entries mentioning a retracted fact."""
        now = datetime.now(tz=timezone.utc)
        # Add a retracted fact
        fact_row = CommittedFactRow(
            id=str(uuid4()),
            candidate_id=str(uuid4()),
            entity_type="person",
            entity_data={"name": "Alice Wonderland"},
            fact_type="observed",
            confidence="high",
            scope="global",
            sensitivity="internal",
            source="test",
            created_at=now,
            valid_from=now,
            belief_status="retracted",
            provenance_chain=[],
        )
        db_session.add(fact_row)

        # Add an episodic entry referencing Alice
        ep_row = EpisodicRow(
            id=str(uuid4()),
            event_id=str(uuid4()),
            memory_type="episodic",
            timestamp=now,
            content="I had a meeting with Alice Wonderland about the project.",
            metadata_extra={},
        )
        db_session.add(ep_row)
        await db_session.flush()

        checker = CrossMemoryChecker(db_session)
        issues = await checker.run()
        ep_issues = [i for i in issues if i["type"] == "episodic_references_retracted_fact"]
        assert len(ep_issues) >= 1

    @pytest.mark.asyncio
    async def test_procedural_references_retracted(self, db_session):
        """Detect procedural entries whose steps reference retracted facts."""
        now = datetime.now(tz=timezone.utc)
        fact_row = CommittedFactRow(
            id=str(uuid4()),
            candidate_id=str(uuid4()),
            entity_type="tool",
            entity_data={"name": "deploy script"},
            fact_type="observed",
            confidence="high",
            scope="global",
            sensitivity="internal",
            source="test",
            created_at=now,
            valid_from=now,
            belief_status="retracted",
            provenance_chain=[],
        )
        db_session.add(fact_row)

        proc_row = ProceduralRow(
            id=str(uuid4()),
            name="deployment",
            description="How to deploy using deploy script",
            steps=["Run deploy script", "Check status"],
            trigger="deploy_request",
            version=1,
            success_rate=0.9,
            metadata_extra={},
        )
        db_session.add(proc_row)
        await db_session.flush()

        checker = CrossMemoryChecker(db_session)
        issues = await checker.run()
        proc_issues = [i for i in issues if i["type"] == "procedural_references_retracted_fact"]
        assert len(proc_issues) >= 1
        assert proc_issues[0]["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_maintenance_includes_consistency(self, db_session):
        """Brain.run_maintenance() returns consistency_issues."""
        brain = await _make_brain(db_session)
        result = await brain.run_maintenance()
        assert "consistency_issues" in result
        assert isinstance(result["consistency_issues"], list)


# ─────────────────────────────────────────────────────────
# Gap 13 — Evolution Category Blocklist
# ─────────────────────────────────────────────────────────

class TestEvolutionBlocklist:
    """Tests for §10.2 forbidden evolution categories."""

    def test_forbidden_categories_defined(self):
        assert "security_boundary" in FORBIDDEN_CATEGORIES
        assert "permission_override" in FORBIDDEN_CATEGORIES
        assert "core_data_transform" in FORBIDDEN_CATEGORIES

    @pytest.mark.asyncio
    async def test_forbidden_category_rejected_at_creation(self, db_session):
        """Proposals for forbidden categories are rejected immediately."""
        sandbox = EvolutionSandbox(db_session, _FakeLLM())
        for cat in FORBIDDEN_CATEGORIES:
            with pytest.raises(ValueError, match="forbidden"):
                await sandbox.propose(
                    change_type=cat,
                    description=f"Attempt {cat}",
                    payload={"test": True},
                )

    @pytest.mark.asyncio
    async def test_allowed_categories_still_work(self, db_session):
        """Non-forbidden categories can still be proposed."""
        sandbox = EvolutionSandbox(db_session, _FakeLLM())
        pid = await sandbox.propose(
            change_type="extraction_prompt",
            description="Improve extraction",
            payload={"prompt": "new prompt"},
        )
        assert pid  # Successfully created

    @pytest.mark.asyncio
    async def test_high_risk_not_forbidden(self, db_session):
        """High-risk categories require approval but are NOT forbidden at creation."""
        sandbox = EvolutionSandbox(db_session, _FakeLLM())
        for cat in ["ontology_extension", "skill_template"]:
            pid = await sandbox.propose(
                change_type=cat,
                description=f"Test {cat}",
                payload={},
            )
            assert pid


# ─────────────────────────────────────────────────────────
# Gap 14 — valid_time in commit_fact
# ─────────────────────────────────────────────────────────

class TestValidTimeInCommitFact:
    """Tests for §16 explicit valid_time support."""

    @pytest.mark.asyncio
    async def test_commit_with_explicit_valid_from(self, db_session):
        """commit_fact honours explicit valid_from."""
        brain = await _make_brain(db_session)
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result = await brain.commit_fact({
            "entity_type": "observation",
            "entity_data": {"description": "historical fact"},
            "valid_from": past,
        })
        assert result["committed"]
        # Verify stored value
        row = await brain.semantic_store.get(result["fact_id"])
        assert row is not None
        assert row.valid_from.year == 2020

    @pytest.mark.asyncio
    async def test_commit_with_explicit_valid_to(self, db_session):
        """commit_fact honours explicit valid_to."""
        brain = await _make_brain(db_session)
        now = datetime.now(tz=timezone.utc)
        future = now + timedelta(days=30)
        result = await brain.commit_fact({
            "entity_type": "observation",
            "entity_data": {"description": "temporary fact"},
            "valid_from": now,
            "valid_to": future,
        })
        row = await brain.semantic_store.get(result["fact_id"])
        assert row.valid_to is not None
        assert row.valid_to > row.valid_from

    @pytest.mark.asyncio
    async def test_commit_without_valid_time_uses_now(self, db_session):
        """Without explicit valid_from, current time is used."""
        brain = await _make_brain(db_session)
        before = datetime.now(tz=timezone.utc)
        result = await brain.commit_fact({
            "entity_type": "observation",
            "entity_data": {"description": "default time"},
        })
        row = await brain.semantic_store.get(result["fact_id"])
        valid_from = row.valid_from
        if valid_from.tzinfo is None:
            valid_from = valid_from.replace(tzinfo=timezone.utc)
        assert valid_from >= before

    @pytest.mark.asyncio
    async def test_schema_accepts_valid_time(self):
        """CommitFactRequest schema includes valid_from / valid_to."""
        from percos.api.schemas import CommitFactRequest
        req = CommitFactRequest(
            entity_type="task",
            entity_data={"name": "test"},
            valid_from=datetime(2023, 6, 1, tzinfo=timezone.utc),
            valid_to=datetime(2023, 12, 31, tzinfo=timezone.utc),
        )
        data = req.model_dump(mode="json")
        assert data["valid_from"] is not None
        assert data["valid_to"] is not None


# ─────────────────────────────────────────────────────────
# Gap 15 — Task / Project Management API
# ─────────────────────────────────────────────────────────

class TestTaskProjectManagement:
    """Tests for schema-driven entity management (formerly §14.3 task/project)."""

    @pytest.mark.asyncio
    async def test_create_entity(self, db_session):
        brain = await _make_brain(db_session)
        result = await brain.create_entity("task", {
            "name": "Write tests",
            "description": "Write unit tests for the module",
            "priority": 2,
            "status": "todo",
        })
        assert result["committed"]
        assert result["entity_id"]

    @pytest.mark.asyncio
    async def test_list_entities(self, db_session):
        brain = await _make_brain(db_session)
        await brain.create_entity("task", {"name": "Task A", "status": "todo"})
        await brain.create_entity("task", {"name": "Task B", "status": "in_progress"})
        await brain.create_entity("task", {"name": "Task C", "status": "done"})

        all_tasks = await brain.list_entities("task")
        assert len(all_tasks) == 3

    @pytest.mark.asyncio
    async def test_update_entity(self, db_session):
        brain = await _make_brain(db_session)
        result = await brain.create_entity("task", {"name": "Original", "status": "todo"})
        entity_id = result["entity_id"]

        update_result = await brain.update_entity(entity_id, {"status": "done"})
        assert update_result.get("updated")

    @pytest.mark.asyncio
    async def test_retract_entity(self, db_session):
        brain = await _make_brain(db_session)
        result = await brain.create_entity("task", {"name": "To Remove"})
        entity_id = result["entity_id"]

        retract_result = await brain.delete_entity(entity_id)
        assert retract_result["retracted"]

        # Verify entity no longer appears in active list
        remaining = await brain.list_entities("task")
        assert all(t["entity_id"] != entity_id for t in remaining)

    @pytest.mark.asyncio
    async def test_create_project_entity(self, db_session):
        brain = await _make_brain(db_session)
        result = await brain.create_entity("project", {
            "name": "PCOS v2",
            "description": "Next generation cognitive OS",
            "status": "todo",
        })
        assert result["committed"]
        assert result["entity_id"]

    @pytest.mark.asyncio
    async def test_list_project_entities(self, db_session):
        brain = await _make_brain(db_session)
        await brain.create_entity("project", {"name": "Project Alpha"})
        await brain.create_entity("project", {"name": "Project Beta"})

        projects = await brain.list_entities("project")
        assert len(projects) == 2

    @pytest.mark.asyncio
    async def test_entity_with_deadline(self, db_session):
        brain = await _make_brain(db_session)
        deadline = datetime(2025, 12, 31, tzinfo=timezone.utc)
        result = await brain.create_entity("task", {
            "name": "Deadline Task",
            "deadline": deadline.isoformat(),
        })
        tasks = await brain.list_entities("task")
        task = next(t for t in tasks if t["entity_id"] == result["entity_id"])
        assert task.get("deadline") is not None

    @pytest.mark.asyncio
    async def test_generic_entity_schemas(self):
        """Generic entity schemas are properly defined."""
        from percos.api.schemas import (
            CreateEntityRequest, EntityResponse, EntityListResponse,
        )
        # GAP-11: entity_type removed from CreateEntityRequest (comes from URL path)
        req = CreateEntityRequest(data={"name": "test"})
        assert req.data == {"name": "test"}

    @pytest.mark.asyncio
    async def test_entity_audit_logged(self, db_session):
        """Entity creation is tracked in audit log."""
        brain = await _make_brain(db_session)
        await brain.create_entity("task", {"name": "Audited Task"})
        entries = await brain.audit_log.query(action="entity_created")
        assert len(entries) >= 1
