"""Self-Evolution Sandbox (§10).

Safe "self-developing" behavior:
- Propose -> Validate -> Simulate -> Score -> Approve -> Deploy

What the system MAY evolve:
  - Extraction prompts, retrieval heuristics, ranking strategies,
    skill templates, ontology extension proposals

What the system MUST NOT change autonomously:
  - Permissions/policy, security boundaries, high-impact action authority,
    core irreversible data transforms
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from percos.llm import LLMClient
from percos.logging import get_logger
from percos.models.enums import ProposalStatus
from percos.models.events import EvolutionProposal
from percos.stores.tables import ProposalRow, EvolutionConfigRow

log = get_logger("evolution")

# Categories that are safe for auto-approval
LOW_RISK_CATEGORIES = {"extraction_prompt", "retrieval_heuristic", "ranking_strategy"}

# Categories that require human approval
HIGH_RISK_CATEGORIES = {"ontology_extension", "policy_change", "skill_template"}

# Categories that MUST NOT be proposed at all (§10.2 – Gap #13)
FORBIDDEN_CATEGORIES = {"security_boundary", "permission_override", "core_data_transform"}

PROPOSAL_VALIDATION_PROMPT = """\
You are a validator for self-evolution proposals in an Ontology-Governed Knowledge Base.
Check if the proposed change is:
1. Schema-consistent (doesn't break existing types/relations)
2. Safe (doesn't affect permissions, security, or critical data)
3. Well-formed (has clear description and payload)

Respond with JSON:
{
  "valid": true/false,
  "issues": ["issue1", "issue2"],
  "risk_level": "low" | "medium" | "high",
  "recommendation": "approve" | "needs_review" | "reject"
}
"""

SCORING_PROMPT = """\
You are a scoring engine for self-evolution proposals.
Given a proposal and its simulation results, score it on:
- accuracy_improvement: 0.0-1.0
- consistency: 0.0-1.0
- safety: 0.0-1.0
- regression_risk: 0.0-1.0 (higher = more risk)

Respond with JSON:
{
  "accuracy_improvement": 0.0,
  "consistency": 0.0,
  "safety": 0.0,
  "regression_risk": 0.0,
  "overall_score": 0.0,
  "recommendation": "deploy" | "needs_review" | "reject"
}
"""


class EvolutionSandbox:
    """Manages the safe self-evolution pipeline."""

    def __init__(self, session: AsyncSession, llm: LLMClient, audit_log=None):
        self._session = session
        self._llm = llm
        self._audit_log = audit_log  # GAP-M8: optional AuditLog instance
        # Deployed configuration parameters that proposals can modify
        self._active_configs: dict[str, Any] = {
            "extraction_prompt_override": None,
            "retrieval_heuristic": "hybrid",
            "ranking_strategy": "recency_confidence",
            "custom_prompts": {},
            "ontology_extensions": [],   # GAP-H8
        }
        self._deployment_history: list[dict[str, Any]] = []
        self._configs_loaded = False

    # ── Config persistence (GAP-H7) ────────────────────
    async def _ensure_configs_loaded(self) -> None:
        """Load persisted configs from DB on first access."""
        if self._configs_loaded:
            return
        stmt = select(EvolutionConfigRow).where(EvolutionConfigRow.id == "active")
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row:
            saved_configs = row.configs or {}
            # Merge saved configs into defaults (preserve new keys)
            for key in self._active_configs:
                if key in saved_configs:
                    self._active_configs[key] = saved_configs[key]
            self._deployment_history = row.deployment_history or []
        self._configs_loaded = True

    async def _persist_configs(self) -> None:
        """Save current configs and deployment history to DB."""
        stmt = select(EvolutionConfigRow).where(EvolutionConfigRow.id == "active")
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row:
            row.configs = dict(self._active_configs)
            row.deployment_history = list(self._deployment_history)
            row.updated_at = datetime.now(tz=timezone.utc)
        else:
            row = EvolutionConfigRow(
                id="active",
                configs=dict(self._active_configs),
                deployment_history=list(self._deployment_history),
            )
            self._session.add(row)
        await self._session.flush()

    # ── 1. Propose ──────────────────────────────────────
    async def propose(
        self,
        change_type: str,
        description: str,
        payload: dict[str, Any],
    ) -> str:
        """Create a new evolution proposal.

        Core interface: propose_evolution(change) -> proposal_id
        Rejects forbidden categories at creation time (§10.2 – Gap #13).
        """
        if change_type in FORBIDDEN_CATEGORIES:
            raise ValueError(
                f"Category '{change_type}' is forbidden and cannot be proposed. "
                f"Forbidden categories: {sorted(FORBIDDEN_CATEGORIES)}"
            )

        proposal = EvolutionProposal(
            change_type=change_type,
            description=description,
            payload=payload,
            status=ProposalStatus.DRAFT.value,
        )
        row = ProposalRow(
            id=str(proposal.id),
            change_type=proposal.change_type,
            description=proposal.description,
            payload=proposal.payload,
            status=proposal.status,
            version=proposal.version,
        )
        self._session.add(row)
        await self._session.flush()
        log.info("proposal_created", proposal_id=str(proposal.id), change_type=change_type)
        # GAP-M8: Audit log
        await self._audit("evolution_propose", proposal_id=str(proposal.id),
                          details={"change_type": change_type, "description": description})
        return str(proposal.id)

    # ── 2. Validate ─────────────────────────────────────
    async def validate(self, proposal_id: str) -> dict[str, Any]:
        """Validate a proposal for schema consistency and safety.

        GAP-H8: For ontology_extension proposals, also checks that proposed
        new entity types don't conflict with existing ones.
        """
        row = await self._get_proposal(proposal_id)
        if not row:
            return {"error": "proposal_not_found"}

        # GAP-H8: Ontology extension validation
        if row.change_type == "ontology_extension":
            ontology_issues = self._validate_ontology_extension(row.payload or {})
            if ontology_issues:
                return {"valid": False, "issues": ontology_issues, "risk_level": "high",
                        "recommendation": "reject"}

        import json
        user_content = (
            f"Proposal: {row.description}\n"
            f"Change type: {row.change_type}\n"
            f"Payload: {json.dumps(row.payload, default=str)}"
        )
        try:
            result = await self._llm.extract_structured(PROPOSAL_VALIDATION_PROMPT, user_content)
        except Exception as exc:
            return {"valid": False, "error": str(exc)}

        if isinstance(result, dict) and result.get("valid"):
            row.status = ProposalStatus.VALIDATED.value
            await self._session.flush()

        log.info("proposal_validated", proposal_id=proposal_id, valid=result.get("valid") if isinstance(result, dict) else False)
        # GAP-M8: Audit log
        valid = result.get("valid") if isinstance(result, dict) else False
        await self._audit("evolution_validate", proposal_id=proposal_id,
                          outcome="success" if valid else "rejected",
                          details={"valid": valid, "change_type": row.change_type,
                                   "issues": result.get("issues", []) if isinstance(result, dict) else []})
        return result if isinstance(result, dict) else {"valid": False}

    @staticmethod
    def _validate_ontology_extension(payload: dict) -> list[str]:
        """GAP-H8: Validate that proposed ontology extensions don't conflict with existing types."""
        from percos.engine.compiler import _get_ontology_model_map
        existing_types = set(_get_ontology_model_map().keys())
        issues = []

        new_entity_types = payload.get("new_entity_types", [])
        for etype in new_entity_types:
            name = etype if isinstance(etype, str) else etype.get("name", "")
            if name.lower() in existing_types:
                issues.append(f"Entity type '{name}' already exists in the ontology")

        return issues

    # ── 3. Simulate ─────────────────────────────────────
    async def simulate(self, proposal_id: str, replay_data: list[dict] | None = None) -> dict[str, Any]:
        """Simulate a proposal against historical data.

        Core interface: simulate_proposal(proposal_id, replay_set) -> eval_report

        If replay_data is provided with raw event content, attempts empirical
        dual-pipeline simulation by running events through:
          1. Baseline extraction (current config)
          2. Proposed extraction (proposal applied)
        Falls back to LLM-estimated comparison when empirical replay isn't possible.
        """
        row = await self._get_proposal(proposal_id)
        if not row:
            return {"error": "proposal_not_found"}

        replay_items = replay_data or []

        # Try empirical replay for extraction/retrieval proposals
        if replay_items and row.change_type in LOW_RISK_CATEGORIES:
            empirical = await self._simulate_empirical(row, replay_items)
            if empirical is not None:
                row.status = ProposalStatus.SIMULATED.value
                row.baseline_score = empirical.get("avg_baseline_score", 0.5)
                await self._session.flush()
                log.info("proposal_simulated", proposal_id=proposal_id,
                         mode="empirical", delta=empirical["improvement_delta"])
                await self._audit("evolution_simulate", proposal_id=proposal_id,
                                  details={"mode": "empirical",
                                           "delta": empirical["improvement_delta"],
                                           "replay_count": len(replay_items)})
                return empirical

        # Fallback: LLM-estimated simulation
        comparisons: list[dict[str, Any]] = []
        baseline_scores: list[float] = []
        proposed_scores: list[float] = []

        for item in replay_items[:20]:  # cap at 20 replays to limit cost
            import json as _json
            sim_prompt = (
                "You are a simulation engine. Compare baseline vs proposed behavior.\n\n"
                f"Historical input: {_json.dumps(item, default=str)}\n\n"
                f"Proposed change: {row.description}\n"
                f"Change payload: {_json.dumps(row.payload, default=str)}\n\n"
                "Respond with JSON:\n"
                '{"baseline_quality": 0.0-1.0, "proposed_quality": 0.0-1.0, '
                '"improvement": true/false, "explanation": "..."}'
            )
            try:
                result = await self._llm.extract_structured(sim_prompt, "Evaluate this change.")
                if isinstance(result, dict):
                    comparisons.append(result)
                    baseline_scores.append(result.get("baseline_quality", 0.5))
                    proposed_scores.append(result.get("proposed_quality", 0.5))
            except Exception as exc:
                comparisons.append({"error": str(exc)})

        avg_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.5
        avg_proposed = sum(proposed_scores) / len(proposed_scores) if proposed_scores else 0.5

        simulation_result = {
            "proposal_id": proposal_id,
            "simulated": True,
            "mode": "llm_estimated",
            "replay_count": len(replay_items),
            "comparisons_run": len(comparisons),
            "avg_baseline_score": round(avg_baseline, 3),
            "avg_proposed_score": round(avg_proposed, 3),
            "improvement_delta": round(avg_proposed - avg_baseline, 3),
            "details": comparisons[:5],  # return first 5 for transparency
        }

        row.status = ProposalStatus.SIMULATED.value
        row.baseline_score = avg_baseline
        await self._session.flush()
        log.info("proposal_simulated", proposal_id=proposal_id,
                 mode="llm_estimated", delta=simulation_result["improvement_delta"])
        # GAP-M8: Audit log
        await self._audit("evolution_simulate", proposal_id=proposal_id,
                          details={"mode": "llm_estimated",
                                   "delta": simulation_result["improvement_delta"],
                                   "replay_count": len(replay_items)})
        return simulation_result

    async def _simulate_empirical(
        self, row: ProposalRow, replay_items: list[dict],
    ) -> dict[str, Any] | None:
        """Run dual-pipeline empirical simulation for extraction proposals.

        Returns None if empirical replay is not possible (missing deps, etc.).
        """
        try:
            from percos.engine.compiler import _get_extraction_prompt
        except ImportError:
            return None

        baseline_prompt = _get_extraction_prompt()

        # Build proposed prompt based on change type
        proposed_prompt = baseline_prompt
        payload = row.payload or {}
        if row.change_type == "extraction_prompt":
            proposed_prompt = payload.get("prompt_override", baseline_prompt)
        elif row.change_type == "ontology_extension":
            extensions = payload.get("new_entity_types", [])
            if extensions:
                type_names = [
                    t if isinstance(t, str) else t.get("name", "")
                    for t in extensions
                ]
                proposed_prompt += (
                    f"\n\nAdditional custom entity types available: {', '.join(type_names)}"
                )

        if proposed_prompt == baseline_prompt:
            # No prompt difference — for retrieval/ranking changes,
            # use LLM-based comparison with change-type-specific evaluation (GAP-8)
            if row.change_type in ("retrieval_heuristic", "ranking_strategy"):
                return await self._simulate_retrieval_ranking(row, replay_items)
            return None

        comparisons: list[dict[str, Any]] = []
        baseline_entity_counts: list[int] = []
        proposed_entity_counts: list[int] = []

        for item in replay_items[:20]:
            content = item.get("content", "")
            if not content:
                continue

            user_content = (
                f"Event type: {item.get('event_type', 'conversation')}\n"
                f"Source: {item.get('source', 'replay')}\n"
                f"Timestamp: {item.get('timestamp', '')}\n"
                f"Content:\n{content}"
            )

            baseline_result: dict = {}
            proposed_result: dict = {}

            # Run baseline extraction
            try:
                baseline_result = await self._llm.extract_structured(
                    baseline_prompt, user_content,
                )
                if not isinstance(baseline_result, dict):
                    baseline_result = {}
            except Exception:
                baseline_result = {}

            # Run proposed extraction
            try:
                proposed_result = await self._llm.extract_structured(
                    proposed_prompt, user_content,
                )
                if not isinstance(proposed_result, dict):
                    proposed_result = {}
            except Exception:
                proposed_result = {}

            b_candidates = baseline_result.get("candidates", [])
            p_candidates = proposed_result.get("candidates", [])
            baseline_entity_counts.append(len(b_candidates))
            proposed_entity_counts.append(len(p_candidates))

            comparisons.append({
                "baseline_entities": len(b_candidates),
                "proposed_entities": len(p_candidates),
                "baseline_types": list({c.get("entity_type", "") for c in b_candidates}),
                "proposed_types": list({c.get("entity_type", "") for c in p_candidates}),
                "delta": len(p_candidates) - len(b_candidates),
            })

        if not comparisons:
            return None

        avg_b = sum(baseline_entity_counts) / len(baseline_entity_counts)
        avg_p = sum(proposed_entity_counts) / len(proposed_entity_counts)
        # Normalise to 0-1 score (more entities ≈ better recall, capped at 1.0)
        max_count = max(max(baseline_entity_counts, default=1), max(proposed_entity_counts, default=1), 1)
        baseline_score = round(avg_b / max_count, 3)
        proposed_score = round(avg_p / max_count, 3)

        return {
            "proposal_id": row.id,
            "simulated": True,
            "mode": "empirical",
            "replay_count": len(replay_items),
            "comparisons_run": len(comparisons),
            "avg_baseline_score": baseline_score,
            "avg_proposed_score": proposed_score,
            "improvement_delta": round(proposed_score - baseline_score, 3),
            "details": comparisons[:5],
        }

    async def _simulate_retrieval_ranking(
        self, row: ProposalRow, replay_items: list[dict],
    ) -> dict[str, Any] | None:
        """Simulate retrieval/ranking strategy changes using LLM-based evaluation (GAP-8).

        Unlike extraction prompt changes, retrieval/ranking changes can't be tested
        by comparing extraction outputs. Instead, we ask the LLM to evaluate relevance
        of retrieval results under baseline vs proposed strategies.
        """
        import json as _json
        comparisons: list[dict[str, Any]] = []
        baseline_scores: list[float] = []
        proposed_scores: list[float] = []

        payload = row.payload or {}
        change_desc = payload.get("description", row.description)
        strategy_before = payload.get("baseline", "default")
        strategy_after = payload.get("proposed", change_desc)

        for item in replay_items[:20]:
            content = item.get("content", "")
            if not content:
                continue

            sim_prompt = (
                f"You are evaluating a {row.change_type} change.\n\n"
                f"User query/event: {content}\n\n"
                f"Baseline strategy: {strategy_before}\n"
                f"Proposed strategy: {strategy_after}\n\n"
                f"Change details: {_json.dumps(payload, default=str)}\n\n"
                "Score both strategies on relevance (0.0-1.0) for this query.\n"
                "Respond with JSON:\n"
                '{"baseline_relevance": 0.0-1.0, "proposed_relevance": 0.0-1.0, '
                '"improvement": true/false, "explanation": "..."}'
            )
            try:
                result = await self._llm.extract_structured(sim_prompt, "Evaluate this change.")
                if isinstance(result, dict):
                    comparisons.append(result)
                    baseline_scores.append(result.get("baseline_relevance", 0.5))
                    proposed_scores.append(result.get("proposed_relevance", 0.5))
            except Exception as exc:
                comparisons.append({"error": str(exc)})

        if not comparisons:
            return None

        avg_b = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.5
        avg_p = sum(proposed_scores) / len(proposed_scores) if proposed_scores else 0.5

        return {
            "proposal_id": row.id,
            "simulated": True,
            "mode": f"empirical_{row.change_type}",
            "replay_count": len(replay_items),
            "comparisons_run": len(comparisons),
            "avg_baseline_score": round(avg_b, 3),
            "avg_proposed_score": round(avg_p, 3),
            "improvement_delta": round(avg_p - avg_b, 3),
            "details": comparisons[:5],
        }

    # ── 4. Score ────────────────────────────────────────
    async def score(self, proposal_id: str, simulation_result: dict | None = None) -> dict[str, Any]:
        """Score a proposal based on simulation results."""
        row = await self._get_proposal(proposal_id)
        if not row:
            return {"error": "proposal_not_found"}

        import json
        user_content = (
            f"Proposal: {row.description}\n"
            f"Change type: {row.change_type}\n"
            f"Simulation: {json.dumps(simulation_result or {}, default=str)}"
        )
        try:
            result = await self._llm.extract_structured(SCORING_PROMPT, user_content)
        except Exception as exc:
            return {"error": str(exc)}

        if isinstance(result, dict):
            row.score = result.get("overall_score")
            row.status = ProposalStatus.SCORED.value
            await self._session.flush()

        log.info("proposal_scored", proposal_id=proposal_id, score=result.get("overall_score") if isinstance(result, dict) else None)
        # GAP-M8: Audit log
        await self._audit("evolution_score", proposal_id=proposal_id,
                          details={"score": result.get("overall_score") if isinstance(result, dict) else None,
                                   "recommendation": result.get("recommendation") if isinstance(result, dict) else None})
        return result if isinstance(result, dict) else {}

    # ── 5. Approve ──────────────────────────────────────
    async def approve(self, proposal_id: str, human_approved: bool = False) -> dict[str, Any]:
        """Approve a proposal (auto or human). Low-risk → auto, high-risk → human required."""
        row = await self._get_proposal(proposal_id)
        if not row:
            return {"error": "proposal_not_found"}

        is_low_risk = row.change_type in LOW_RISK_CATEGORIES

        if is_low_risk or human_approved:
            row.status = ProposalStatus.APPROVED.value
            await self._session.flush()
            log.info("proposal_approved", proposal_id=proposal_id, auto=is_low_risk)
            # GAP-M8: Audit log
            await self._audit("evolution_approve", proposal_id=proposal_id,
                              details={"auto_approved": is_low_risk, "change_type": row.change_type})
            return {"approved": True, "auto_approved": is_low_risk}

        return {
            "approved": False,
            "reason": "High-risk change requires human approval",
            "change_type": row.change_type,
        }

    # ── 6. Deploy ───────────────────────────────────────
    async def deploy(self, proposal_id: str) -> dict[str, Any]:
        """Deploy an approved proposal – actually apply payload to runtime config.

        Core interface: promote_proposal(proposal_id) -> deployment_version

        GAP-H7: Persists configs to DB after deployment.
        GAP-H8: Handles ontology_extension change types.
        """
        await self._ensure_configs_loaded()
        row = await self._get_proposal(proposal_id)
        if not row:
            return {"error": "proposal_not_found"}
        if row.status != ProposalStatus.APPROVED.value:
            return {"error": f"Proposal not approved (status: {row.status})"}

        # Save snapshot for rollback
        snapshot = {k: v for k, v in self._active_configs.items()}
        self._deployment_history.append({
            "proposal_id": proposal_id,
            "snapshot_before": snapshot,
            "deployed_at": datetime.now(tz=timezone.utc).isoformat(),
        })

        # Apply payload to active config
        payload = row.payload or {}

        # GAP-H8: Handle ontology extension proposals
        if row.change_type == "ontology_extension":
            extensions = self._active_configs.get("ontology_extensions", [])
            new_types = payload.get("new_entity_types", [])
            new_relations = payload.get("new_relation_types", [])
            extensions.append({
                "proposal_id": proposal_id,
                "entity_types": new_types,
                "relation_types": new_relations,
                "deployed_at": datetime.now(tz=timezone.utc).isoformat(),
            })
            self._active_configs["ontology_extensions"] = extensions
        else:
            for key, value in payload.items():
                if key in self._active_configs:
                    self._active_configs[key] = value

        row.status = ProposalStatus.DEPLOYED.value
        row.version += 1
        await self._session.flush()

        # GAP-H7: Persist configs to DB
        await self._persist_configs()

        log.info("proposal_deployed", proposal_id=proposal_id, version=row.version,
                 applied_keys=list(payload.keys()))
        # GAP-M8: Audit log
        await self._audit("evolution_deploy", proposal_id=proposal_id,
                          details={"version": row.version, "change_type": row.change_type,
                                   "applied_keys": list(payload.keys())})
        return {
            "deployed": True,
            "proposal_id": proposal_id,
            "version": row.version,
            "change_type": row.change_type,
            "applied_config_keys": list(payload.keys()),
        }

    # ── Rollback ────────────────────────────────────────
    async def rollback(self, proposal_id: str) -> dict[str, Any]:
        """Rollback a deployed proposal – restore config from snapshot.

        GAP-H7: Persists configs to DB after rollback.
        """
        await self._ensure_configs_loaded()
        row = await self._get_proposal(proposal_id)
        if not row:
            return {"error": "proposal_not_found"}

        # Find the deployment snapshot and restore
        restored = False
        for entry in reversed(self._deployment_history):
            if entry["proposal_id"] == proposal_id:
                self._active_configs.update(entry["snapshot_before"])
                restored = True
                break

        row.status = ProposalStatus.ROLLED_BACK.value
        await self._session.flush()

        # GAP-H7: Persist configs to DB
        await self._persist_configs()

        log.info("proposal_rolled_back", proposal_id=proposal_id, config_restored=restored)
        # GAP-M8: Audit log
        await self._audit("evolution_rollback", proposal_id=proposal_id,
                          details={"config_restored": restored, "change_type": row.change_type})
        return {"rolled_back": True, "proposal_id": proposal_id, "config_restored": restored}

    async def get_active_config(self) -> dict[str, Any]:
        """Return the current active configuration set (GAP-H7: loads from DB)."""
        await self._ensure_configs_loaded()
        return dict(self._active_configs)

    # ── List proposals ──────────────────────────────────
    async def list_proposals(self, status: str | None = None) -> list[dict]:
        conditions = []
        if status:
            conditions.append(ProposalRow.status == status)
        stmt = select(ProposalRow).order_by(ProposalRow.created_at.desc())
        if conditions:
            stmt = stmt.where(*conditions)
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        return [
            {
                "id": r.id,
                "change_type": r.change_type,
                "description": r.description,
                "status": r.status,
                "score": r.score,
                "version": r.version,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    # ── Full pipeline ───────────────────────────────────
    async def run_pipeline(
        self,
        change_type: str,
        description: str,
        payload: dict[str, Any],
        auto_approve: bool = False,
    ) -> dict[str, Any]:
        """Run the full evolution pipeline: propose → validate → simulate → score → approve → deploy."""
        proposal_id = await self.propose(change_type, description, payload)
        validation = await self.validate(proposal_id)
        if not validation.get("valid"):
            return {"proposal_id": proposal_id, "status": "rejected_at_validation", "validation": validation}

        simulation = await self.simulate(proposal_id)
        scoring = await self.score(proposal_id, simulation)

        is_low_risk = change_type in LOW_RISK_CATEGORIES
        approval = await self.approve(proposal_id, human_approved=auto_approve)

        if approval.get("approved"):
            deployment = await self.deploy(proposal_id)
            return {"proposal_id": proposal_id, "status": "deployed", "deployment": deployment}

        return {"proposal_id": proposal_id, "status": "awaiting_approval", "scoring": scoring}

    async def _get_proposal(self, proposal_id: str) -> ProposalRow | None:
        stmt = select(ProposalRow).where(ProposalRow.id == proposal_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # ── GAP-M8: Audit helper ───────────────────────────
    async def _audit(
        self,
        action: str,
        proposal_id: str,
        outcome: str = "success",
        details: dict | None = None,
    ) -> None:
        """Record an evolution audit entry if an audit_log is available."""
        if self._audit_log is None:
            return
        try:
            await self._audit_log.record(
                action, "evolution",
                resource_id=proposal_id, resource_type="proposal",
                outcome=outcome,
                details=details or {},
            )
        except Exception:
            log.debug("evolution_audit_failed", action=action, proposal_id=proposal_id)
