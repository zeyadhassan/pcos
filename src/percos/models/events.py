"""Event & memory candidate models for the ingestion / compilation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .enums import (
    CandidateRouting,
    Confidence,
    EventType,
    FactType,
    MemoryType,
    ProposalStatus,
    Sensitivity,
)


# ── Raw Event (§7.2 input) ──────────────────────────────
class RawEvent(BaseModel):
    """An incoming event from any source (chat, calendar, tool, etc.)."""
    id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    source: str = ""                          # e.g. "user_chat", "google_calendar"
    content: str = ""                         # raw text / JSON payload
    metadata_extra: dict[str, Any] = Field(default_factory=dict)


# ── Candidate Fact (§7.3 output) ────────────────────────
class CandidateFact(BaseModel):
    """A potential knowledge item extracted by the Memory Compiler."""
    id: UUID = Field(default_factory=uuid4)
    event_id: UUID                            # source event
    entity_type: str = ""                     # ontology type name
    entity_data: dict[str, Any] = Field(default_factory=dict)
    relation_type: str | None = None
    relation_source_id: UUID | None = None
    relation_target_id: UUID | None = None
    fact_type: FactType = FactType.OBSERVED
    confidence: Confidence = Confidence.MEDIUM
    scope: str = "global"
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    routing: CandidateRouting = CandidateRouting.NEEDS_VERIFICATION
    provenance: list[str] = Field(default_factory=list)
    conflicts_with: list[UUID] = Field(default_factory=list)
    raw_extraction: str = ""                  # LLM-generated extraction text


# ── Committed Fact ──────────────────────────────────────
class CommittedFact(BaseModel):
    """A fact that has been accepted into the semantic memory store."""
    id: UUID = Field(default_factory=uuid4)
    candidate_id: UUID
    entity_type: str
    entity_data: dict[str, Any] = Field(default_factory=dict)
    fact_type: FactType
    confidence: Confidence
    scope: str = "global"
    sensitivity: Sensitivity
    source: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    valid_from: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    valid_to: datetime | None = None
    last_verified: datetime | None = None
    provenance_chain: list[str] = Field(default_factory=list)


# ── Episodic Memory Entry ───────────────────────────────
class EpisodicEntry(BaseModel):
    """An entry in episodic (append-only) memory."""
    id: UUID = Field(default_factory=uuid4)
    event_id: UUID
    memory_type: MemoryType = MemoryType.EPISODIC
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    content: str = ""
    embedding: list[float] | None = None
    metadata_extra: dict[str, Any] = Field(default_factory=dict)


# ── Procedural Memory Entry ────────────────────────────
class ProceduralEntry(BaseModel):
    """A skill / workflow / template."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    steps: list[str] = Field(default_factory=list)
    trigger: str = ""
    version: int = 1
    success_rate: float = 0.0
    metadata_extra: dict[str, Any] = Field(default_factory=dict)


# ── Working Memory Snapshot ─────────────────────────────
class WorkingMemory(BaseModel):
    """Current session state."""
    active_goals: list[UUID] = Field(default_factory=list)
    current_plan: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    session_context: dict[str, Any] = Field(default_factory=dict)


# ── Policy Memory Entry ────────────────────────────────
class PolicyEntry(BaseModel):
    """A permission / safety / privacy rule."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    rule: str
    effect: str = "deny"          # allow | deny | require_approval
    priority: int = 0
    scope: str = "global"
    active: bool = True


# ── Context Bundle (§9 output) ─────────────────────────
class ContextBundle(BaseModel):
    """Assembled retrieval context passed to the planner."""
    query: str
    semantic_facts: list[CommittedFact] = Field(default_factory=list)
    episodic_entries: list[EpisodicEntry] = Field(default_factory=list)
    policies: list[PolicyEntry] = Field(default_factory=list)
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)
    graph_context: dict[str, Any] = Field(default_factory=dict)
    relevance_scores: dict[str, float] = Field(default_factory=dict)  # fact_id → relevance score


# ── Evolution Proposal (§10.3) ─────────────────────────
class EvolutionProposal(BaseModel):
    """A candidate change proposed by the self-evolution layer."""
    id: UUID = Field(default_factory=uuid4)
    change_type: str = ""          # e.g. "extraction_prompt", "retrieval_heuristic"
    description: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    status: ProposalStatus = ProposalStatus.DRAFT
    score: float | None = None
    baseline_score: float | None = None
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
