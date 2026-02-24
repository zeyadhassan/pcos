"""SQLAlchemy ORM tables for relational / semantic memory."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ── Raw Events ──────────────────────────────────────────
class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type: Mapped[str] = mapped_column(String(50))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    source: Mapped[str] = mapped_column(String(200), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    metadata_extra: Mapped[dict] = mapped_column(JSON, default=dict)


# ── Candidate Facts ─────────────────────────────────────
class CandidateFactRow(Base):
    __tablename__ = "candidate_facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id: Mapped[str] = mapped_column(String(36), ForeignKey("events.id"), index=True)
    entity_type: Mapped[str] = mapped_column(String(100), default="", index=True)
    entity_data: Mapped[dict] = mapped_column(JSON, default=dict)
    relation_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    relation_source_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    relation_target_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    fact_type: Mapped[str] = mapped_column(String(30), default="observed")
    confidence: Mapped[str] = mapped_column(String(20), default="medium")
    scope: Mapped[str] = mapped_column(String(20), default="global")
    sensitivity: Mapped[str] = mapped_column(String(20), default="internal")
    routing: Mapped[str] = mapped_column(String(30), default="needs_verification", index=True)
    provenance: Mapped[list] = mapped_column(JSON, default=list)
    conflicts_with: Mapped[list] = mapped_column(JSON, default=list)
    raw_extraction: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))


# ── Committed Facts (Semantic Memory) ───────────────────
class CommittedFactRow(Base):
    __tablename__ = "committed_facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id: Mapped[str] = mapped_column(String(36), ForeignKey("candidate_facts.id"), index=True)
    entity_type: Mapped[str] = mapped_column(String(100), index=True)
    entity_data: Mapped[dict] = mapped_column(JSON, default=dict)
    fact_type: Mapped[str] = mapped_column(String(30))
    confidence: Mapped[str] = mapped_column(String(20))
    scope: Mapped[str] = mapped_column(String(20))
    sensitivity: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    last_verified: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    valid_from: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    belief_status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    provenance_chain: Mapped[list] = mapped_column(JSON, default=list)


# ── Episodic Memory ─────────────────────────────────────
class EpisodicRow(Base):
    __tablename__ = "episodic_memory"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id: Mapped[str] = mapped_column(String(36), ForeignKey("events.id"), index=True)
    memory_type: Mapped[str] = mapped_column(String(20), default="episodic")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    content: Mapped[str] = mapped_column(Text, default="")
    metadata_extra: Mapped[dict] = mapped_column(JSON, default=dict)


# ── Procedural Memory ──────────────────────────────────
class ProceduralRow(Base):
    __tablename__ = "procedural_memory"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    steps: Mapped[list] = mapped_column(JSON, default=list)
    trigger: Mapped[str] = mapped_column(String(300), default="")
    version: Mapped[int] = mapped_column(Integer, default=1)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_extra: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))


# ── Procedural Version History ─────────────────────────
class ProceduralVersionRow(Base):
    __tablename__ = "procedural_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    procedure_id: Mapped[str] = mapped_column(String(36))
    version: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    steps: Mapped[list] = mapped_column(JSON, default=list)
    trigger: Mapped[str] = mapped_column(String(300), default="")
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_extra: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))


# ── Policy Memory ──────────────────────────────────────
class PolicyRow(Base):
    __tablename__ = "policy_memory"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(200))
    rule: Mapped[str] = mapped_column(Text)
    effect: Mapped[str] = mapped_column(String(30), default="deny")
    priority: Mapped[int] = mapped_column(Integer, default=0)
    scope: Mapped[str] = mapped_column(String(20), default="global")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))


# ── Knowledge Graph Relations ──────────────────────────
class RelationRow(Base):
    __tablename__ = "relations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id: Mapped[str] = mapped_column(String(36), ForeignKey("committed_facts.id"), index=True)
    target_id: Mapped[str] = mapped_column(String(36), ForeignKey("committed_facts.id"), index=True)
    relation_type: Mapped[str] = mapped_column(String(100), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    fact_type: Mapped[str] = mapped_column(String(30), default="observed")
    confidence: Mapped[str] = mapped_column(String(20), default="medium")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    valid_from: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    provenance_chain: Mapped[list] = mapped_column(JSON, default=list)


# ── Evolution Proposals ────────────────────────────────
class ProposalRow(Base):
    __tablename__ = "evolution_proposals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    change_type: Mapped[str] = mapped_column(String(100), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="draft")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))


# ── Working Memory Persistence (Gap #18) ───────────────
class WorkingMemoryRow(Base):
    __tablename__ = "working_memory"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default="default")
    state: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))


# ── Evaluation Metrics (§13) ────────────────────────────
class MetricRow(Base):
    __tablename__ = "evaluation_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(200), index=True)
    value: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc), index=True)
    metadata_extra: Mapped[dict] = mapped_column(JSON, default=dict)


# ── Audit Log (§11.2, §17 principle 8) ─────────────────
class AuditLogRow(Base):
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
    action: Mapped[str] = mapped_column(String(100))             # e.g. "fact_committed", "belief_deleted"
    component: Mapped[str] = mapped_column(String(50))           # e.g. "compiler", "brain", "evolution"
    actor: Mapped[str] = mapped_column(String(200), default="")  # user, system, source
    resource_id: Mapped[str] = mapped_column(String(36), default="")  # affected entity ID
    resource_type: Mapped[str] = mapped_column(String(100), default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    outcome: Mapped[str] = mapped_column(String(30), default="success")  # success | failure | blocked


# ── Evolution Config Persistence (GAP-H7) ──────────────
class EvolutionConfigRow(Base):
    __tablename__ = "evolution_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default="active")
    configs: Mapped[dict] = mapped_column(JSON, default=dict)
    deployment_history: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(tz=timezone.utc))
