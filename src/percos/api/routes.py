"""API routes – the REST surface of the Personal Cognitive OS."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from percos.api.deps import get_brain
from percos.api.schemas import (
    BeliefsResponse,
    BeliefHistoryResponse,
    ChatRequest,
    ChatResponse,
    CommitFactRequest,
    CommitFactResponse,
    CompileRequest,
    CompileResponse,
    CreateEntityRequest,
    EntityDetailResponse,
    EntityListResponse,
    EntityResponse,
    ExecuteActionRequest,
    ExecuteActionResponse,
    ExplainResponse,
    ExportResponse,
    HealthResponse,
    IngestEventRequest,
    IngestEventResponse,
    MaintenanceResponse,
    MetricsResponse,
    PendingCandidatesResponse,
    PlanActionRequest,
    PlanActionResponse,
    ProposeRequest,
    ProposeResponse,
    ProposalListResponse,
    SimulateRequest,
    QueryRequest,
    QueryResponse,
    ReflectRequest,
    ReflectResponse,
    ResetRequest,
    ResetResponse,
    SchemaResponse,
    SuggestionsResponse,
    UpdateBeliefRequest,
    ValidateRequest,
    ValidateResponse,
)
from percos.engine.brain import Brain
from percos.models.events import RawEvent
from percos.models.enums import EventType

router = APIRouter()


# ── Health ──────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health():
    from percos import __version__
    return HealthResponse(status="ok", version=__version__)


# ── Domain Schema (Domain-Agnostic Framework) ──────────

@router.get("/schema")
async def get_schema():
    """Return the active domain schema definition."""
    from percos.schema import get_domain_schema
    schema = get_domain_schema()
    return schema.to_dict()


# ── Generic Entity CRUD (Domain-Agnostic) ──────────────

@router.post("/entities/{entity_type}", response_model=EntityResponse)
async def create_entity(
    entity_type: str,
    req: CreateEntityRequest,
    brain: Brain = Depends(get_brain),
):
    """Create an entity of any type defined in the domain schema."""
    result = await brain.create_entity(entity_type, req.data)
    return EntityResponse(**result)


@router.get("/entities/{entity_type}", response_model=EntityListResponse)
async def list_entities(
    entity_type: str,
    brain: Brain = Depends(get_brain),
):
    """List all active entities of a given type."""
    entities = await brain.list_entities(entity_type)
    return EntityListResponse(
        entities=entities,
        entity_type=entity_type,
        count=len(entities),
    )


@router.get("/entities/{entity_type}/{entity_id}", response_model=EntityDetailResponse)
async def get_entity(
    entity_type: str,
    entity_id: str,
    brain: Brain = Depends(get_brain),
):
    """Get a single entity by type and ID."""
    entity = await brain.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return EntityDetailResponse(entity=entity)


@router.put("/entities/{entity_type}/{entity_id}")
async def update_entity(
    entity_type: str,
    entity_id: str,
    req: UpdateBeliefRequest,
    brain: Brain = Depends(get_brain),
):
    """Update an entity's data fields."""
    updates = req.model_dump(exclude_none=True)
    entity_updates = updates.get("entity_data", updates)
    result = await brain.update_entity(entity_id, entity_updates)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.delete("/entities/{entity_type}/{entity_id}")
async def delete_entity(
    entity_type: str,
    entity_id: str,
    brain: Brain = Depends(get_brain),
):
    """Delete (retract) an entity."""
    return await brain.delete_entity(entity_id)


# ── Chat (full pipeline) ───────────────────────────────
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, brain: Brain = Depends(get_brain)):
    result = await brain.chat(req.message, req.source)
    return ChatResponse(**result)


# ── Event Ingestion ─────────────────────────────────────
@router.post("/events/ingest", response_model=IngestEventResponse)
async def ingest_event(req: IngestEventRequest, brain: Brain = Depends(get_brain)):
    event = RawEvent(
        event_type=EventType(req.event_type),
        content=req.content,
        source=req.source,
        metadata_extra=req.metadata_extra,
    )
    event_id = await brain.ingest_event(event)
    await brain.session.commit()
    return IngestEventResponse(event_id=event_id)


# ── Memory Compilation ──────────────────────────────────
@router.post("/memory/compile", response_model=CompileResponse)
async def compile_memory(req: CompileRequest, brain: Brain = Depends(get_brain)):
    candidates = await brain.compile_memory(req.event_id)
    await brain.session.commit()
    return CompileResponse(candidates=candidates)


# ── Candidate Validation ────────────────────────────────
@router.post("/memory/validate", response_model=ValidateResponse)
async def validate_candidate(req: ValidateRequest, brain: Brain = Depends(get_brain)):
    result = await brain.validate_candidate(req.candidate_id, req.accept)
    await brain.session.commit()
    return ValidateResponse(**result)


# ── World Model Query ───────────────────────────────────
@router.post("/query", response_model=QueryResponse)
async def query_world_model(req: QueryRequest, brain: Brain = Depends(get_brain)):
    bundle = await brain.query_world_model(req.query, task_type=req.task_type)
    return QueryResponse(
        query=bundle.query,
        semantic_facts=[f.model_dump(mode="json") for f in bundle.semantic_facts],
        episodic_entries=[e.model_dump(mode="json") for e in bundle.episodic_entries],
        policies=[p.model_dump(mode="json") for p in bundle.policies],
        graph_context=bundle.graph_context,
        relevance_scores=bundle.relevance_scores,
    )


# ── Standalone plan / execute / reflect (§16) ───────────
@router.post("/plan", response_model=PlanActionResponse)
async def plan_action(req: PlanActionRequest, brain: Brain = Depends(get_brain)):
    context = await brain.query_world_model(req.query, task_type=req.task_type)
    plan = await brain.plan_action(req.query, context)
    return PlanActionResponse(plan=plan)


@router.post("/execute", response_model=ExecuteActionResponse)
async def execute_action(req: ExecuteActionRequest, brain: Brain = Depends(get_brain)):
    query = req.query or req.plan.get("goal", "")
    context = await brain.query_world_model(query, task_type=req.task_type)
    outcome = await brain.execute_action(req.plan, context)
    return ExecuteActionResponse(outcome=outcome)


@router.post("/reflect", response_model=ReflectResponse)
async def reflect_on_outcome(req: ReflectRequest, brain: Brain = Depends(get_brain)):
    reflection = await brain.reflect_on_outcome(req.outcome)
    return ReflectResponse(reflection=reflection)


# ── Beliefs (Memory Control Panel §12) ──────────────────
@router.get("/beliefs", response_model=BeliefsResponse)
async def list_beliefs(entity_type: str | None = None, brain: Brain = Depends(get_brain)):
    beliefs = await brain.get_beliefs(entity_type)
    return BeliefsResponse(beliefs=beliefs)


@router.get("/beliefs/{fact_id}/explain", response_model=ExplainResponse)
async def explain_belief(fact_id: str, brain: Brain = Depends(get_brain)):
    provenance = await brain.explain_belief(fact_id)
    if "error" in provenance:
        raise HTTPException(status_code=404, detail=provenance["error"])
    return ExplainResponse(provenance=provenance)


@router.get("/beliefs/history/{entity_name}", response_model=BeliefHistoryResponse)
async def belief_history(entity_name: str, brain: Brain = Depends(get_brain)):
    history = await brain.get_belief_history(entity_name)
    return BeliefHistoryResponse(history=history)


@router.put("/beliefs/{fact_id}")
async def update_belief(fact_id: str, req: UpdateBeliefRequest, brain: Brain = Depends(get_brain)):
    updates = req.model_dump(exclude_none=True)
    result = await brain.update_belief(fact_id, updates)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.delete("/beliefs/{fact_id}")
async def delete_belief(fact_id: str, brain: Brain = Depends(get_brain)):
    result = await brain.delete_belief(fact_id)
    return result


# ── Evolution Proposals (§10) ───────────────────────────
@router.post("/evolution/propose", response_model=ProposeResponse)
async def propose_evolution(req: ProposeRequest, brain: Brain = Depends(get_brain)):
    proposal_id = await brain.propose_evolution(req.model_dump())
    await brain.session.commit()
    return ProposeResponse(proposal_id=proposal_id)


@router.get("/evolution/proposals", response_model=ProposalListResponse)
async def list_proposals(status: str | None = None, brain: Brain = Depends(get_brain)):
    proposals = await brain.evolution.list_proposals(status)
    return ProposalListResponse(proposals=proposals)


@router.post("/evolution/{proposal_id}/validate")
async def validate_proposal(proposal_id: str, brain: Brain = Depends(get_brain)):
    result = await brain.evolution.validate(proposal_id)
    await brain.session.commit()
    return result


@router.post("/evolution/{proposal_id}/simulate")
async def simulate_proposal(
    proposal_id: str,
    body: SimulateRequest | None = None,
    brain: Brain = Depends(get_brain),
):
    replay = body.replay_set if body else []
    result = await brain.evolution.simulate(proposal_id, replay_data=replay)
    await brain.session.commit()
    return result


@router.post("/evolution/{proposal_id}/approve")
async def approve_proposal(proposal_id: str, human_approved: bool = False, brain: Brain = Depends(get_brain)):
    result = await brain.evolution.approve(proposal_id, human_approved)
    await brain.session.commit()
    return result


@router.post("/evolution/{proposal_id}/deploy")
async def deploy_proposal(proposal_id: str, brain: Brain = Depends(get_brain)):
    result = await brain.evolution.deploy(proposal_id)
    await brain.session.commit()
    return result


@router.post("/evolution/{proposal_id}/rollback")
async def rollback_proposal(proposal_id: str, brain: Brain = Depends(get_brain)):
    result = await brain.evolution.rollback(proposal_id)
    await brain.session.commit()
    return result


# ── Evaluation Metrics (§13) ────────────────────────────
@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(brain: Brain = Depends(get_brain)):
    """Retrieve aggregated evaluation metrics."""
    metrics = await brain.get_metrics()
    return MetricsResponse(metrics=metrics)


# ── Maintenance ─────────────────────────────────────────
@router.post("/maintenance/run", response_model=MaintenanceResponse)
async def run_maintenance(brain: Brain = Depends(get_brain)):
    result = await brain.run_maintenance()
    return MaintenanceResponse(**result)


# ── Direct Fact Commit (Gap #6) ─────────────────────────
@router.post("/facts/commit", response_model=CommitFactResponse)
async def commit_fact(req: CommitFactRequest, brain: Brain = Depends(get_brain)):
    result = await brain.commit_fact(req.model_dump())
    return CommitFactResponse(**result)


# ── Pending Candidates (Gap #7) ────────────────────────
@router.get("/candidates/pending", response_model=PendingCandidatesResponse)
async def list_pending_candidates(brain: Brain = Depends(get_brain)):
    candidates = await brain.list_pending_candidates()
    return PendingCandidatesResponse(candidates=candidates)


# ── Quarantined Candidates (GAP-H2) ───────────────────
@router.get("/candidates/quarantined", response_model=PendingCandidatesResponse)
async def list_quarantined_candidates(brain: Brain = Depends(get_brain)):
    """List candidates in quarantine for review."""
    candidates = await brain.list_quarantined_candidates()
    return PendingCandidatesResponse(candidates=candidates)


# ── Export / Reset (Gap #8) ────────────────────────────
@router.get("/memory/export", response_model=ExportResponse)
async def export_memory(brain: Brain = Depends(get_brain)):
    data = await brain.export_memory()
    return ExportResponse(**data)


@router.post("/memory/reset", response_model=ResetResponse)
async def reset_memory(req: ResetRequest, brain: Brain = Depends(get_brain)):
    result = await brain.reset_memory(confirm=req.confirm)
    return ResetResponse(**result)


# ── Audit Log (§11.2) ─────────────────────────────────
@router.get("/audit")
async def query_audit_log(
    action: str | None = None,
    component: str | None = None,
    resource_id: str | None = None,
    actor: str | None = None,
    limit: int = 100,
    brain: Brain = Depends(get_brain),
):
    entries = await brain.audit_log.query(
        action=action, component=component,
        resource_id=resource_id, actor=actor, limit=limit,
    )
    return {"entries": entries, "count": len(entries)}

# ── Document Import (GAP-H10) ───────────────────────
from percos.api.schemas import DocumentImportRequest, DocumentImportResponse

@router.post("/documents/import", response_model=DocumentImportResponse)
async def import_document(
    body: DocumentImportRequest,
    brain: Brain = Depends(get_brain),
):
    """Import a document (plain text or markdown) into the memory system.

    The document is chunked and each chunk is ingested as a DOCUMENT event,
    then compiled through the memory compiler.
    """
    result = await brain.import_document(
        content=body.content, source=body.source, title=body.title, content_type=body.content_type,
    )
    return DocumentImportResponse(**result)


# ── Suggestions (§14.3) ───────────────────────────────
@router.get("/suggestions", response_model=SuggestionsResponse)
async def get_suggestions(brain: Brain = Depends(get_brain)):
    """Return prioritised next-action recommendations."""
    suggestions = await brain.suggest_next_actions()
    return SuggestionsResponse(suggestions=suggestions)


# ── Deadlines & Reminders (GAP-M2) ───────────────────
@router.get("/deadlines")
async def check_deadlines(horizon_days: int = 7, brain: Brain = Depends(get_brain)):
    """Return upcoming deadlines within the given horizon."""
    deadlines = await brain.deadline_checker.check_upcoming_deadlines(horizon_days)
    return {"deadlines": deadlines, "count": len(deadlines)}


@router.get("/reminders")
async def get_reminders(horizon_days: int = 7, brain: Brain = Depends(get_brain)):
    """Return actionable reminders for upcoming/overdue items."""
    reminders = await brain.deadline_checker.get_reminders(horizon_days)
    return {"reminders": reminders, "count": len(reminders)}


# ── Procedural Memory (GAP-H4) ───────────────────────
from percos.api.schemas import ProcedureListResponse, ProcedureHistoryResponse

@router.get("/procedures", response_model=ProcedureListResponse)
async def list_procedures(brain: Brain = Depends(get_brain)):
    """List all procedural memory entries (skills/workflows)."""
    rows = await brain.procedural_store.list_all()
    procedures = [
        {
            "id": r.id, "name": r.name, "description": r.description,
            "steps": r.steps, "trigger": r.trigger, "version": r.version,
            "success_rate": r.success_rate,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
    return ProcedureListResponse(procedures=procedures)


@router.get("/procedures/{procedure_id}/history", response_model=ProcedureHistoryResponse)
async def procedure_history(procedure_id: str, brain: Brain = Depends(get_brain)):
    """Get version history for a procedure."""
    history = await brain.procedural_store.get_version_history(procedure_id)
    return ProcedureHistoryResponse(history=history)


# ── Policy CRUD (GAP-H4) ────────────────────────────
from percos.api.schemas import (
    CreatePolicyRequest, UpdatePolicyRequest,
    PolicyListResponse, PolicyResponse,
)
from percos.models.events import PolicyEntry

@router.get("/policies", response_model=PolicyListResponse)
async def list_policies(brain: Brain = Depends(get_brain)):
    """List all policies."""
    rows = await brain.policy_store.get_active_policies()
    policies = [
        {
            "id": r.id, "name": r.name, "rule": r.rule,
            "effect": r.effect, "priority": r.priority,
            "scope": r.scope, "active": r.active,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
    return PolicyListResponse(policies=policies)


@router.post("/policies", response_model=PolicyResponse)
async def create_policy(req: CreatePolicyRequest, brain: Brain = Depends(get_brain)):
    """Create a new policy."""
    entry = PolicyEntry(
        name=req.name, rule=req.rule, effect=req.effect,
        priority=req.priority, scope=req.scope,
    )
    policy_id = await brain.policy_store.save(entry)
    await brain.audit_log.record(
        "policy_created", "brain", actor="user",
        resource_id=policy_id, resource_type="policy",
        details={"name": req.name},
    )
    await brain.session.commit()
    return PolicyResponse(policy_id=policy_id)


@router.put("/policies/{policy_id}")
async def update_policy(policy_id: str, req: UpdatePolicyRequest, brain: Brain = Depends(get_brain)):
    """Update a policy (enable/disable, edit rule, etc.)."""
    row = await brain.policy_store.get(policy_id)
    if not row:
        raise HTTPException(status_code=404, detail="Policy not found")
    updates = req.model_dump(exclude_none=True)
    for field, value in updates.items():
        if hasattr(row, field):
            setattr(row, field, value)
    await brain.audit_log.record(
        "policy_updated", "brain", actor="user",
        resource_id=policy_id, resource_type="policy",
        details={"fields_updated": list(updates.keys())},
    )
    await brain.session.commit()
    return {"updated": True, "policy_id": policy_id}


@router.delete("/policies/{policy_id}")
async def delete_policy(policy_id: str, brain: Brain = Depends(get_brain)):
    """Delete a policy."""
    from sqlalchemy import delete as sql_delete
    from percos.stores.tables import PolicyRow
    row = await brain.policy_store.get(policy_id)
    if not row:
        raise HTTPException(status_code=404, detail="Policy not found")
    await brain.session.delete(row)
    await brain.audit_log.record(
        "policy_deleted", "brain", actor="user",
        resource_id=policy_id, resource_type="policy",
    )
    await brain.session.commit()
    return {"deleted": True, "policy_id": policy_id}


# ── GAP-L1: External Integrations ───────────────────
from percos.api.schemas import (
    AdapterListResponse, AdapterTestResponse,
    SyncRequest, SyncResponse, SyncAllResponse,
)

@router.get("/integrations/adapters", response_model=AdapterListResponse)
async def list_adapters(brain: Brain = Depends(get_brain)):
    """List all registered external integration adapters."""
    return AdapterListResponse(adapters=brain.integration_manager.list_adapters())


@router.post("/integrations/test", response_model=AdapterTestResponse)
async def test_adapters(brain: Brain = Depends(get_brain)):
    """Test connectivity for all registered adapters."""
    results = await brain.integration_manager.test_all()
    return AdapterTestResponse(results=results)


@router.post("/integrations/sync", response_model=SyncResponse)
async def sync_adapter(req: SyncRequest, brain: Brain = Depends(get_brain)):
    """Sync events from a specific external adapter."""
    result = await brain.integration_manager.sync(
        req.adapter_name, since=req.since, limit=req.limit,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    await brain.session.commit()
    return SyncResponse(**result)


@router.post("/integrations/sync-all", response_model=SyncAllResponse)
async def sync_all_adapters(brain: Brain = Depends(get_brain)):
    """Sync events from all registered adapters."""
    results = await brain.integration_manager.sync_all()
    await brain.session.commit()
    return SyncAllResponse(results=results)


# ── GAP-L2: Multi-modal Ingestion ───────────────────
from percos.api.schemas import MultiModalIngestResponse

@router.post("/events/ingest-multimodal", response_model=MultiModalIngestResponse)
async def ingest_multimodal(
    content: str,
    content_type: str,
    source: str = "multimodal",
    brain: Brain = Depends(get_brain),
):
    """Ingest multi-modal content (voice, image, structured data).

    ``content`` should be base64-encoded for binary formats,
    or plain text for structured data (JSON, CSV).
    """
    event_id, extracted = await brain.ingestion.ingest_multimodal(
        content_bytes=content,
        content_type=content_type,
        source=source,
    )
    await brain.session.commit()
    return MultiModalIngestResponse(
        event_id=event_id,
        content_type=content_type,
        extracted_text=extracted[:500],  # cap preview
    )


# ── GAP-L3: Privacy / Redaction ─────────────────────
from percos.api.schemas import (
    RedactionRuleRequest, RedactionRuleResponse, RedactedTextResponse,
)
from percos.engine.security import get_redaction_engine

@router.post("/redaction/rules", response_model=RedactionRuleResponse)
async def add_redaction_rule(req: RedactionRuleRequest):
    """Add a custom redaction rule."""
    engine = get_redaction_engine()
    engine.add_rule(
        name=req.name,
        pattern=req.pattern,
        replacement=req.replacement,
        sensitivity_level=req.sensitivity_level,
    )
    return RedactionRuleResponse(name=req.name)


@router.get("/redaction/rules")
async def list_redaction_rules():
    """List all active redaction rules."""
    engine = get_redaction_engine()
    return {"rules": engine.list_rules()}


@router.delete("/redaction/rules/{name}")
async def remove_redaction_rule(name: str):
    """Remove a redaction rule."""
    engine = get_redaction_engine()
    removed = engine.remove_rule(name)
    if not removed:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"removed": True, "name": name}


@router.post("/redaction/redact", response_model=RedactedTextResponse)
async def redact_text(
    text: str,
    clearance: str = "internal",
):
    """Redact PII and sensitive content from text."""
    engine = get_redaction_engine()
    redacted, rules = engine.redact(text, requester_clearance=clearance)
    return RedactedTextResponse(
        original_length=len(text),
        redacted_length=len(redacted),
        redacted_text=redacted,
        rules_applied=rules,
    )


@router.post("/redaction/detect")
async def detect_pii(text: str):
    """Detect PII in text without redacting."""
    engine = get_redaction_engine()
    findings = engine.detect_pii(text)
    return {"findings": findings, "count": len(findings)}


# ── GAP-L4: Cross-device Sync ──────────────────────
from percos.api.schemas import (
    SyncExportResponse, SyncImportRequest, SyncImportResponse,
)
from percos.engine.sync import SyncProtocol, get_device_id

@router.get("/sync/export", response_model=SyncExportResponse)
async def sync_export(
    since: str | None = None,
    include_episodic: bool = True,
    brain: Brain = Depends(get_brain),
):
    """Export cognitive state for cross-device synchronisation."""
    from datetime import datetime as dt
    since_dt = dt.fromisoformat(since) if since else None
    protocol = SyncProtocol(brain.session)
    data = await protocol.export_state(since=since_dt, include_episodic=include_episodic)
    return SyncExportResponse(
        facts=data["facts"],
        episodic=data["episodic"],
        procedures=data["procedures"],
        policies=data["policies"],
        version_vector=data["version_vector"],
        exported_at=data["exported_at"],
    )


@router.post("/sync/import", response_model=SyncImportResponse)
async def sync_import(req: SyncImportRequest, brain: Brain = Depends(get_brain)):
    """Import cognitive state from another device."""
    protocol = SyncProtocol(brain.session)
    result = await protocol.import_state(
        payload=req.model_dump(),
        strategy=req.strategy,
    )
    await brain.session.commit()
    return SyncImportResponse(**result)


@router.get("/sync/device-id")
async def get_sync_device_id():
    """Return the unique device identifier for this instance."""
    return {"device_id": get_device_id()}


# ── GAP-L5: Identity Resolution ────────────────────
from percos.api.schemas import (
    DuplicatesResponse, MergeEntitiesRequest, MergeEntitiesResponse,
)
from percos.engine.identity_resolution import IdentityResolver

@router.get("/entities/duplicates", response_model=DuplicatesResponse)
async def find_duplicates(
    entity_type: str | None = None,
    threshold: float = 0.85,
    brain: Brain = Depends(get_brain),
):
    """Scan committed facts for potential duplicate entities."""
    resolver = IdentityResolver(brain.session, threshold=threshold)
    dupes = await resolver.find_duplicates(entity_type=entity_type, threshold=threshold)
    return DuplicatesResponse(duplicates=dupes)


@router.post("/entities/merge", response_model=MergeEntitiesResponse)
async def merge_entities(req: MergeEntitiesRequest, brain: Brain = Depends(get_brain)):
    """Merge duplicate entities into a canonical fact."""
    resolver = IdentityResolver(brain.session)
    result = await resolver.merge(req.canonical_id, req.duplicate_ids)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    await brain.session.commit()
    return MergeEntitiesResponse(**result)


# ── GAP-L6: Metrics History ────────────────────────
from percos.api.schemas import MetricsHistoryResponse

@router.get("/metrics/history", response_model=MetricsHistoryResponse)
async def get_metrics_history(
    metric_name: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 500,
    brain: Brain = Depends(get_brain),
):
    """Query historical metric samples with optional filters."""
    from datetime import datetime as dt
    since_dt = dt.fromisoformat(since) if since else None
    until_dt = dt.fromisoformat(until) if until else None
    samples = await brain.evaluation.query_history(
        brain.session,
        metric_name=metric_name,
        since=since_dt,
        until=until_dt,
        limit=limit,
    )
    return MetricsHistoryResponse(samples=samples, count=len(samples))

