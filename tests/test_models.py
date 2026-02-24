"""Tests for core domain models."""

from percos.models.enums import (
    BeliefStatus,
    CandidateRouting,
    Confidence,
    EventType,
    FactType,
    IntentType,
    MemoryType,
    ProposalStatus,
    Sensitivity,
)
from percos.models.ontology import (
    Entity,
    FactMetadata,
    Relation,
)
from percos.models.events import (
    RawEvent,
    CandidateFact,
    CommittedFact,
    EpisodicEntry,
    ProceduralEntry,
    WorkingMemory,
    PolicyEntry,
    ContextBundle,
    EvolutionProposal,
)


class TestEnums:
    def test_fact_types(self):
        assert FactType.OBSERVED.value == "observed"
        assert FactType.DERIVED.value == "derived"
        assert FactType.HYPOTHESIS.value == "hypothesis"
        assert FactType.POLICY.value == "policy"

    def test_confidence_levels(self):
        assert Confidence.HIGH.value == "high"
        assert Confidence.MEDIUM.value == "medium"
        assert Confidence.LOW.value == "low"

    def test_event_types(self):
        assert EventType.CONVERSATION.value == "conversation"
        assert EventType.EXTERNAL.value == "external"

    def test_candidate_routing(self):
        assert CandidateRouting.AUTO_ACCEPT.value == "auto_accept"
        assert CandidateRouting.QUARANTINE.value == "quarantine"


class TestOntologyModels:
    def test_create_entity(self):
        e = Entity(name="Alice")
        assert e.name == "Alice"
        assert e.id is not None
        assert e.description == ""

    def test_fact_metadata_defaults(self):
        meta = FactMetadata()
        assert meta.confidence == Confidence.MEDIUM
        assert meta.sensitivity == Sensitivity.INTERNAL
        assert meta.scope == "global"
        assert meta.belief_status == BeliefStatus.ACTIVE

    def test_relation(self):
        import uuid
        r = Relation(
            source_id=uuid.uuid4(),
            target_id=uuid.uuid4(),
            relation_type="depends_on",
        )
        assert r.relation_type == "depends_on"
        assert r.weight == 1.0


class TestEventModels:
    def test_raw_event(self):
        event = RawEvent(
            event_type=EventType.CONVERSATION,
            source="user_chat",
            content="Hello, world",
        )
        assert event.event_type == EventType.CONVERSATION
        assert event.content == "Hello, world"

    def test_candidate_fact(self):
        import uuid
        cf = CandidateFact(
            event_id=uuid.uuid4(),
            entity_type="preference",
            entity_data={"name": "coffee", "value": "black"},
            fact_type=FactType.OBSERVED,
            confidence=Confidence.HIGH,
        )
        assert cf.entity_type == "preference"
        assert cf.routing == CandidateRouting.NEEDS_VERIFICATION

    def test_working_memory(self):
        wm = WorkingMemory()
        assert wm.active_goals == []
        assert wm.current_plan == []

    def test_context_bundle(self):
        bundle = ContextBundle(query="What are my tasks?")
        assert bundle.query == "What are my tasks?"
        assert bundle.semantic_facts == []

    def test_evolution_proposal(self):
        ep = EvolutionProposal(
            change_type="extraction_prompt",
            description="Improve entity extraction accuracy",
        )
        assert ep.status == "draft"
        assert ep.version == 1
