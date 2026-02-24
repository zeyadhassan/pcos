"""Tests for the cognitive engine components (ingestion, compiler routing, TTM, runtime)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from percos.engine.ingestion import EventIngestion
from percos.engine.compiler import MemoryCompiler
from percos.engine.ttm import TemporalTruthMaintenance
from percos.models.enums import (
    CandidateRouting,
    Confidence,
    EventType,
    FactType,
    Sensitivity,
)
from percos.models.events import CandidateFact, CommittedFact, RawEvent
from percos.stores.episodic_store import EpisodicStore
from percos.stores.semantic_store import SemanticStore
from percos.stores.tables import CommittedFactRow


class TestEventIngestion:
    @pytest.mark.asyncio
    async def test_ingest_event(self, db_session: AsyncSession):
        episodic = EpisodicStore(db_session)
        ingestion = EventIngestion(db_session, episodic)

        event = RawEvent(
            event_type=EventType.CONVERSATION,
            source="test",
            content="I prefer to have meetings in the morning.",
        )
        event_id = await ingestion.ingest(event)
        assert event_id is not None

        # Check episodic entry was created
        entries = await episodic.list_recent(limit=1)
        assert len(entries) == 1
        assert "meetings in the morning" in entries[0].content


class TestCompilerRouting:
    """Test the routing logic of the memory compiler (no LLM needed)."""

    def test_high_confidence_observed_auto_accept(self):
        candidate = CandidateFact(
            event_id=uuid.uuid4(),
            entity_type="preference",
            fact_type=FactType.OBSERVED,
            confidence=Confidence.HIGH,
            conflicts_with=[],
        )
        compiler = MemoryCompiler.__new__(MemoryCompiler)
        routing = compiler._route(candidate)
        assert routing == CandidateRouting.AUTO_ACCEPT

    def test_hypothesis_needs_verification(self):
        candidate = CandidateFact(
            event_id=uuid.uuid4(),
            entity_type="preference",
            fact_type=FactType.HYPOTHESIS,
            confidence=Confidence.MEDIUM,
        )
        compiler = MemoryCompiler.__new__(MemoryCompiler)
        routing = compiler._route(candidate)
        assert routing == CandidateRouting.NEEDS_VERIFICATION

    def test_policy_needs_user_confirm(self):
        candidate = CandidateFact(
            event_id=uuid.uuid4(),
            entity_type="policy",
            fact_type=FactType.POLICY,
            confidence=Confidence.HIGH,
        )
        compiler = MemoryCompiler.__new__(MemoryCompiler)
        routing = compiler._route(candidate)
        assert routing == CandidateRouting.NEEDS_USER_CONFIRM

    def test_conflicts_need_user_confirm(self):
        candidate = CandidateFact(
            event_id=uuid.uuid4(),
            entity_type="preference",
            fact_type=FactType.OBSERVED,
            confidence=Confidence.HIGH,
            conflicts_with=[uuid.uuid4()],
        )
        compiler = MemoryCompiler.__new__(MemoryCompiler)
        routing = compiler._route(candidate)
        assert routing == CandidateRouting.NEEDS_USER_CONFIRM

    def test_secret_sensitivity_quarantine(self):
        candidate = CandidateFact(
            event_id=uuid.uuid4(),
            entity_type="observation",
            fact_type=FactType.OBSERVED,
            confidence=Confidence.MEDIUM,
            sensitivity=Sensitivity.SECRET,
        )
        compiler = MemoryCompiler.__new__(MemoryCompiler)
        routing = compiler._route(candidate)
        assert routing == CandidateRouting.QUARANTINE


class TestTemporalTruthMaintenance:
    @pytest.mark.asyncio
    async def test_detect_stale_facts(self, db_session: AsyncSession):
        # Insert an old fact
        old_fact = CommittedFactRow(
            id=str(uuid.uuid4()),
            candidate_id=str(uuid.uuid4()),
            entity_type="preference",
            entity_data={"name": "old_pref"},
            fact_type="observed",
            confidence="medium",
            scope="global",
            sensitivity="internal",
            belief_status="active",
            created_at=datetime.now(tz=timezone.utc) - timedelta(days=100),
            last_verified=datetime.now(tz=timezone.utc) - timedelta(days=100),
            valid_from=datetime.now(tz=timezone.utc) - timedelta(days=100),
            provenance_chain=[],
        )
        db_session.add(old_fact)
        await db_session.flush()

        # Mock LLM – TTM doesn't need LLM for staleness detection
        class MockLLM:
            pass

        ttm = TemporalTruthMaintenance(db_session, MockLLM())  # type: ignore
        stale = await ttm.detect_stale_facts(days=90)
        assert len(stale) >= 1

    @pytest.mark.asyncio
    async def test_mark_stale(self, db_session: AsyncSession):
        fact = CommittedFactRow(
            id=str(uuid.uuid4()),
            candidate_id=str(uuid.uuid4()),
            entity_type="observation",
            entity_data={"name": "test"},
            fact_type="observed",
            confidence="low",
            scope="global",
            sensitivity="public",
            belief_status="active",
            provenance_chain=[],
        )
        db_session.add(fact)
        await db_session.flush()

        class MockLLM:
            pass

        ttm = TemporalTruthMaintenance(db_session, MockLLM())  # type: ignore
        count = await ttm.mark_stale([fact.id])
        assert count == 1

    @pytest.mark.asyncio
    async def test_supersede(self, db_session: AsyncSession):
        old_id = str(uuid.uuid4())
        new_id = str(uuid.uuid4())
        old_fact = CommittedFactRow(
            id=old_id,
            candidate_id=str(uuid.uuid4()),
            entity_type="preference",
            entity_data={"name": "meeting_time", "value": "afternoon"},
            fact_type="observed",
            confidence="medium",
            scope="work",
            sensitivity="internal",
            belief_status="active",
            provenance_chain=[],
        )
        new_fact = CommittedFactRow(
            id=new_id,
            candidate_id=str(uuid.uuid4()),
            entity_type="preference",
            entity_data={"name": "meeting_time", "value": "morning"},
            fact_type="observed",
            confidence="high",
            scope="work",
            sensitivity="internal",
            belief_status="active",
            provenance_chain=[],
        )
        db_session.add(old_fact)
        db_session.add(new_fact)
        await db_session.flush()

        class MockLLM:
            pass

        ttm = TemporalTruthMaintenance(db_session, MockLLM())  # type: ignore
        await ttm.supersede(old_id, new_id)

        from sqlalchemy import select
        from percos.stores.tables import CommittedFactRow as CFR
        stmt = select(CFR).where(CFR.id == old_id)
        result = await db_session.execute(stmt)
        row = result.scalar_one()
        assert row.belief_status == "superseded"
