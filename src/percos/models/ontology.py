"""Ontology entity definitions – framework base classes for the knowledge graph.

Concrete entity types are defined in the domain schema YAML file and generated
dynamically at runtime by the Domain Schema System (``percos.schema``).
This module provides only the framework-level base classes:

  * ``Entity``        – base for all ontology nodes
  * ``FactMetadata``  – temporal & provenance metadata
  * ``Relation``      – typed directed edge
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .enums import (
    BeliefStatus,
    Confidence,
    FactType,
    Sensitivity,
)


# ── Temporal & Provenance Mixin (§6.3) ──────────────────
class FactMetadata(BaseModel):
    """Required metadata for every fact / belief (§6.3)."""
    source: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    valid_from: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    valid_to: datetime | None = None
    confidence: Confidence = Confidence.MEDIUM
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    scope: str = "global"
    last_verified: datetime | None = None
    provenance_chain: list[str] = Field(default_factory=list)
    fact_type: FactType = FactType.OBSERVED
    belief_status: BeliefStatus = BeliefStatus.ACTIVE


# ── Base Entity ──────────────────────────────────────────
class Entity(BaseModel):
    """Base class for all ontology entities."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    metadata: FactMetadata = Field(default_factory=FactMetadata)
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


# ── Relation (generic typed edge) ──────────────────────
class Relation(BaseModel):
    """A typed directed edge in the knowledge graph."""
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    relation_type: str
    weight: float = 1.0
    metadata: FactMetadata = Field(default_factory=FactMetadata)
