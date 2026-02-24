"""Tests for memory stores (DB-backed)."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from percos.models.enums import Confidence, FactType, MemoryType, Sensitivity
from percos.models.events import (
    CommittedFact,
    EpisodicEntry,
    PolicyEntry,
    ProceduralEntry,
)
from percos.stores.episodic_store import EpisodicStore
from percos.stores.semantic_store import SemanticStore
from percos.stores.procedural_policy_stores import PolicyStore, ProceduralStore
from percos.stores.graph import KnowledgeGraph
from percos.stores.tables import CommittedFactRow, RelationRow


class TestEpisodicStore:
    @pytest.mark.asyncio
    async def test_append_and_get(self, db_session: AsyncSession):
        store = EpisodicStore(db_session)
        entry = EpisodicEntry(
            event_id=uuid.uuid4(),
            content="User said hello",
        )
        entry_id = await store.append(entry)
        assert entry_id == str(entry.id)

        row = await store.get(entry_id)
        assert row is not None
        assert row.content == "User said hello"

    @pytest.mark.asyncio
    async def test_list_recent(self, db_session: AsyncSession):
        store = EpisodicStore(db_session)
        for i in range(5):
            entry = EpisodicEntry(
                event_id=uuid.uuid4(),
                content=f"Episode {i}",
            )
            await store.append(entry)

        recent = await store.list_recent(limit=3)
        assert len(recent) == 3


class TestSemanticStore:
    @pytest.mark.asyncio
    async def test_commit_and_get(self, db_session: AsyncSession):
        store = SemanticStore(db_session)
        fact = CommittedFact(
            candidate_id=uuid.uuid4(),
            entity_type="preference",
            entity_data={"name": "coffee", "value": "black"},
            fact_type=FactType.OBSERVED,
            confidence=Confidence.HIGH,
            scope="personal",
            sensitivity=Sensitivity.INTERNAL,
            source="user_chat",
        )
        fact_id = await store.commit(fact)
        assert fact_id is not None

        row = await store.get(fact_id)
        assert row is not None
        assert row.entity_type == "preference"

    @pytest.mark.asyncio
    async def test_get_active_facts(self, db_session: AsyncSession):
        store = SemanticStore(db_session)
        fact = CommittedFact(
            candidate_id=uuid.uuid4(),
            entity_type="task",
            entity_data={"name": "Write tests"},
            fact_type=FactType.OBSERVED,
            confidence=Confidence.MEDIUM,
            scope="work",
            sensitivity=Sensitivity.INTERNAL,
        )
        await store.commit(fact)

        facts = await store.get_active_facts("task")
        assert len(facts) >= 1

    @pytest.mark.asyncio
    async def test_supersede(self, db_session: AsyncSession):
        store = SemanticStore(db_session)
        old_fact = CommittedFact(
            candidate_id=uuid.uuid4(),
            entity_type="preference",
            entity_data={"name": "meeting_time", "value": "afternoons"},
            fact_type=FactType.OBSERVED,
            confidence=Confidence.MEDIUM,
            scope="work",
            sensitivity=Sensitivity.INTERNAL,
        )
        old_id = await store.commit(old_fact)

        new_fact = CommittedFact(
            candidate_id=uuid.uuid4(),
            entity_type="preference",
            entity_data={"name": "meeting_time", "value": "mornings"},
            fact_type=FactType.OBSERVED,
            confidence=Confidence.HIGH,
            scope="work",
            sensitivity=Sensitivity.INTERNAL,
        )
        new_id = await store.supersede(old_id, new_fact)

        old_row = await store.get(old_id)
        assert old_row.belief_status == "superseded"

    @pytest.mark.asyncio
    async def test_retract(self, db_session: AsyncSession):
        store = SemanticStore(db_session)
        fact = CommittedFact(
            candidate_id=uuid.uuid4(),
            entity_type="observation",
            entity_data={"name": "temp_note"},
            fact_type=FactType.HYPOTHESIS,
            confidence=Confidence.LOW,
            scope="global",
            sensitivity=Sensitivity.PUBLIC,
        )
        fact_id = await store.commit(fact)
        await store.retract(fact_id)

        row = await store.get(fact_id)
        assert row.belief_status == "retracted"

    @pytest.mark.asyncio
    async def test_explain(self, db_session: AsyncSession):
        store = SemanticStore(db_session)
        fact = CommittedFact(
            candidate_id=uuid.uuid4(),
            entity_type="preference",
            entity_data={"name": "theme", "value": "dark"},
            fact_type=FactType.OBSERVED,
            confidence=Confidence.HIGH,
            scope="global",
            sensitivity=Sensitivity.PUBLIC,
            provenance_chain=["event:abc", "llm_extraction"],
        )
        fact_id = await store.commit(fact)
        explanation = await store.explain(fact_id)
        assert explanation["fact_id"] == fact_id
        assert "llm_extraction" in explanation["provenance_chain"]


class TestKnowledgeGraph:
    def test_add_and_query(self):
        kg = KnowledgeGraph()
        kg.add_node("n1", entity_type="person", data={"name": "Alice"})
        kg.add_node("n2", entity_type="project", data={"name": "PCOS"})
        kg.add_edge("n1", "n2", relation_type="works_on")

        assert kg.node_count == 2
        assert kg.edge_count == 1

        neighborhood = kg.neighbors("n1", depth=1)
        assert len(neighborhood["nodes"]) >= 1

    def test_find_by_type(self):
        kg = KnowledgeGraph()
        kg.add_node("p1", entity_type="person", data={"name": "Alice"})
        kg.add_node("p2", entity_type="person", data={"name": "Bob"})
        kg.add_node("t1", entity_type="task", data={"name": "Test"})

        people = kg.find_by_type("person")
        assert len(people) == 2

    def test_shortest_path(self):
        kg = KnowledgeGraph()
        kg.add_node("a")
        kg.add_node("b")
        kg.add_node("c")
        kg.add_edge("a", "b")
        kg.add_edge("b", "c")

        path = kg.shortest_path("a", "c")
        assert path == ["a", "b", "c"]


class TestPolicyStore:
    @pytest.mark.asyncio
    async def test_save_and_check(self, db_session: AsyncSession):
        store = PolicyStore(db_session)
        policy = PolicyEntry(
            name="no_delete_projects",
            rule="Do not delete any project without explicit user approval",
            effect="deny",
            priority=10,
        )
        await store.save(policy)

        result = await store.check_action("delete", "project")
        assert result["effect"] == "deny"

    @pytest.mark.asyncio
    async def test_default_allow(self, db_session: AsyncSession):
        store = PolicyStore(db_session)
        result = await store.check_action("read", "document")
        assert result["allowed"] is True


class TestProceduralStore:
    @pytest.mark.asyncio
    async def test_save_and_list(self, db_session: AsyncSession):
        store = ProceduralStore(db_session)
        entry = ProceduralEntry(
            name="Daily standup",
            description="Run daily standup routine",
            steps=["Gather updates", "Review blockers", "Plan day"],
            trigger="morning",
        )
        await store.save(entry)

        all_entries = await store.list_all()
        assert len(all_entries) >= 1

        found = await store.find_by_trigger("morning")
        assert len(found) >= 1
