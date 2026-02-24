"""API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Chat ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    source: str = "user_chat"

class ChatResponse(BaseModel):
    event_id: str
    response: str
    candidates_extracted: int
    plan: dict | None = None
    reflection: dict | None = None


# ── Event Ingestion ─────────────────────────────────────
class IngestEventRequest(BaseModel):
    event_type: str         # conversation, document, tool_result, external
    content: str
    source: str = ""
    metadata_extra: dict[str, Any] = Field(default_factory=dict)

class IngestEventResponse(BaseModel):
    event_id: str


# ── Memory Compilation ─────────────────────────────────
class CompileRequest(BaseModel):
    event_id: str

class CompileResponse(BaseModel):
    candidates: list[dict]


# ── Candidate Validation ───────────────────────────────
class ValidateRequest(BaseModel):
    candidate_id: str
    accept: bool = True

class ValidateResponse(BaseModel):
    accepted: bool
    fact_id: str | None = None


# ── World Model Query ──────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    task_type: str | None = None  # recall | planning | execution | reflection

class QueryResponse(BaseModel):
    query: str
    semantic_facts: list[dict]
    episodic_entries: list[dict]
    policies: list[dict]
    graph_context: dict
    relevance_scores: dict[str, float] = Field(default_factory=dict)  # fact_id → score


# ── Standalone plan / execute / reflect (§16) ──────────
class PlanActionRequest(BaseModel):
    query: str
    task_type: str | None = None

class PlanActionResponse(BaseModel):
    plan: dict

class ExecuteActionRequest(BaseModel):
    plan: dict
    query: str | None = None
    task_type: str | None = None

class ExecuteActionResponse(BaseModel):
    outcome: dict

class ReflectRequest(BaseModel):
    outcome: dict

class ReflectResponse(BaseModel):
    reflection: dict


# ── Beliefs (Memory Control Panel) ─────────────────────
class BeliefsResponse(BaseModel):
    beliefs: list[dict]

class UpdateBeliefRequest(BaseModel):
    entity_data: dict[str, Any] | None = None
    confidence: str | None = None
    scope: str | None = None
    sensitivity: str | None = None

class BeliefHistoryResponse(BaseModel):
    history: list[dict]


# ── Evolution Proposals ────────────────────────────────
class ProposeRequest(BaseModel):
    change_type: str
    description: str
    payload: dict[str, Any] = Field(default_factory=dict)

class ProposeResponse(BaseModel):
    proposal_id: str

class ProposalListResponse(BaseModel):
    proposals: list[dict]

class SimulateRequest(BaseModel):
    """Optional body for POST /evolution/{id}/simulate."""
    replay_set: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Historical events to replay against the proposal for A/B comparison",
    )


# ── Maintenance ────────────────────────────────────────
class MaintenanceResponse(BaseModel):
    stale_detected: int
    marked_stale: int
    contradictions: list[dict] = Field(default_factory=list)
    consistency_issues: list[dict] = Field(default_factory=list)


# ── Evaluation Metrics (§13) ───────────────────────────
class MetricsResponse(BaseModel):
    metrics: dict[str, Any] = Field(default_factory=dict)


# ── Direct Fact Commit (Gap #6) ────────────────────────
class CommitFactRequest(BaseModel):
    entity_type: str
    entity_data: dict[str, Any] = Field(default_factory=dict)
    fact_type: str = "observed"
    confidence: str = "high"
    scope: str = "global"
    sensitivity: str = "internal"
    source: str = "user_direct"
    valid_from: datetime | None = None   # explicit temporal anchor (§16 – Gap #14)
    valid_to: datetime | None = None

class CommitFactResponse(BaseModel):
    fact_id: str
    committed: bool


# ── Pending Candidates (Gap #7) ────────────────────────
class PendingCandidatesResponse(BaseModel):
    candidates: list[dict]


# ── Export / Reset (Gap #8) ────────────────────────────
class ExportResponse(BaseModel):
    semantic_facts: list[dict]
    episodic_entries: list[dict]
    procedural_entries: list[dict]
    policies: list[dict]

class ResetRequest(BaseModel):
    confirm: bool = False

class ResetResponse(BaseModel):
    reset: bool = False
    error: str | None = None


# ── Explain ────────────────────────────────────────────
class ExplainResponse(BaseModel):
    provenance: dict


# ── General ────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""


# ── Generic Entity CRUD (Domain-Agnostic) ─────────────
class CreateEntityRequest(BaseModel):
    """Create an entity of any type defined in the domain schema."""
    data: dict[str, Any] = Field(default_factory=dict)

class EntityResponse(BaseModel):
    entity_id: str
    entity_type: str
    committed: bool = True

class EntityListResponse(BaseModel):
    entities: list[dict] = Field(default_factory=list)
    entity_type: str = ""
    count: int = 0

class EntityDetailResponse(BaseModel):
    entity: dict | None = None

class SchemaResponse(BaseModel):
    """Response for GET /schema — the active domain schema definition."""
    schema_def: dict[str, Any] = Field(default_factory=dict, alias="schema")

    model_config = ConfigDict(populate_by_name=True)


# ── Suggestions (§14.3) ───────────────────────────────
class SuggestionsResponse(BaseModel):
    suggestions: list[dict] = Field(default_factory=list)


# ── Procedural Memory (GAP-H4) ───────────────────────
class ProcedureListResponse(BaseModel):
    procedures: list[dict]

class ProcedureHistoryResponse(BaseModel):
    history: list[dict]


# ── Policy CRUD (GAP-H4) ────────────────────────────
class CreatePolicyRequest(BaseModel):
    name: str
    rule: str
    effect: str = "deny"
    priority: int = 0
    scope: str = "global"

class UpdatePolicyRequest(BaseModel):
    name: str | None = None
    rule: str | None = None
    effect: str | None = None
    priority: int | None = None
    scope: str | None = None
    active: bool | None = None

class PolicyListResponse(BaseModel):
    policies: list[dict]

class PolicyResponse(BaseModel):
    policy_id: str
    created: bool = True


# ── Document Import (GAP-H10) ───────────────────────

class DocumentImportRequest(BaseModel):
    content: str
    source: str = "document_import"
    title: str | None = None
    content_type: str = "text/plain"

class DocumentImportResponse(BaseModel):
    event_ids: list[str]
    chunks: int
    title: str | None = None


# ── External Integrations (GAP-L1) ───────────────────
class AdapterListResponse(BaseModel):
    adapters: list[dict]

class AdapterTestResponse(BaseModel):
    results: dict[str, dict]

class SyncRequest(BaseModel):
    adapter_name: str
    since: datetime | None = None
    limit: int = 50

class SyncResponse(BaseModel):
    adapter: str
    fetched: int
    ingested: int
    event_ids: list[str] = Field(default_factory=list)

class SyncAllResponse(BaseModel):
    results: dict[str, Any] = Field(default_factory=dict)


# ── Metrics History (GAP-L6) ─────────────────────────
class MetricsHistoryRequest(BaseModel):
    metric_name: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 500

class MetricsHistoryResponse(BaseModel):
    samples: list[dict] = Field(default_factory=list)
    count: int = 0


# ── Multi-modal Ingestion (GAP-L2) ──────────────────
class MultiModalIngestResponse(BaseModel):
    event_id: str
    content_type: str
    extracted_text: str = ""


# ── Privacy / Redaction (GAP-L3) ─────────────────────
class RedactionRuleRequest(BaseModel):
    name: str
    pattern: str          # regex pattern to match
    replacement: str = "[REDACTED]"
    sensitivity_level: str = "private"   # minimum sensitivity to apply

class RedactionRuleResponse(BaseModel):
    name: str
    added: bool = True

class RedactedTextResponse(BaseModel):
    original_length: int
    redacted_length: int
    redacted_text: str
    rules_applied: list[str] = Field(default_factory=list)


# ── Cross-device Sync (GAP-L4) ──────────────────────
class SyncExportResponse(BaseModel):
    facts: list[dict] = Field(default_factory=list)
    episodic: list[dict] = Field(default_factory=list)
    procedures: list[dict] = Field(default_factory=list)
    policies: list[dict] = Field(default_factory=list)
    version_vector: dict[str, int] = Field(default_factory=dict)
    exported_at: datetime | None = None

class SyncImportRequest(BaseModel):
    facts: list[dict] = Field(default_factory=list)
    procedures: list[dict] = Field(default_factory=list)
    policies: list[dict] = Field(default_factory=list)
    strategy: str = "newest_wins"  # newest_wins | manual | merge

class SyncImportResponse(BaseModel):
    imported: int = 0
    skipped: int = 0
    conflicts: list[dict] = Field(default_factory=list)


# ── Identity Resolution (GAP-L5) ────────────────────
class DuplicatesResponse(BaseModel):
    duplicates: list[dict] = Field(default_factory=list)

class MergeEntitiesRequest(BaseModel):
    canonical_id: str
    duplicate_ids: list[str]

class MergeEntitiesResponse(BaseModel):
    merged: int = 0
    canonical_id: str = ""
