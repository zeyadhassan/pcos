"""Brain – the top-level orchestrator that ties all cognitive components together.

Implements the core API surface (§16):
  ingest_event, compile_memory, validate_candidate, commit_fact,
  query_world_model, plan_action, execute_action, reflect_on_outcome,
  propose_evolution, simulate_proposal, promote_proposal, explain_belief
"""

from __future__ import annotations

import json as _json
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from percos.engine.compiler import MemoryCompiler
from percos.engine.evaluation import get_evaluation_harness
from percos.engine.evolution import EvolutionSandbox
from percos.engine.ingestion import EventIngestion
from percos.engine.retrieval import RetrievalPlanner
from percos.engine.runtime import CognitiveRuntime
from percos.engine.ttm import TemporalTruthMaintenance
from percos.llm import LLMClient
from percos.logging import get_logger
from percos.models.enums import BeliefStatus, Confidence, FactType, Sensitivity
from percos.models.events import CommittedFact, ContextBundle, RawEvent, WorkingMemory
from percos.stores.episodic_store import EpisodicStore
from percos.stores.graph import KnowledgeGraph
from percos.stores.procedural_policy_stores import PolicyStore, ProceduralStore
from percos.stores.semantic_store import SemanticStore
from percos.stores.tables import CandidateFactRow, CommittedFactRow, RelationRow, WorkingMemoryRow
from percos.stores.audit_log import AuditLog
from percos.engine.style_tracker import CommunicationStyleTracker
from percos.engine.consistency import CrossMemoryChecker
from percos.engine.deadlines import DeadlineChecker
from percos.engine.integrations import IntegrationManager

log = get_logger("brain")


class Brain:
    """Top-level orchestrator – the Personal Cognitive OS 'brain'."""

    def __init__(
        self,
        session: AsyncSession,
        llm: LLMClient,
        chroma_collection=None,
    ):
        self.session = session
        self.llm = llm
        self._chroma = chroma_collection

        # Stores
        self.episodic_store = EpisodicStore(session, chroma_collection)
        self.semantic_store = SemanticStore(session, chroma_collection)
        self.procedural_store = ProceduralStore(session)
        self.policy_store = PolicyStore(session)
        self.knowledge_graph = KnowledgeGraph()

        # Engine components
        self.ingestion = EventIngestion(session, self.episodic_store)
        self.compiler = MemoryCompiler(session, llm, self.semantic_store, evolution_sandbox=None)
        self.ttm = TemporalTruthMaintenance(session, llm)
        self.retrieval = RetrievalPlanner(
            llm, self.semantic_store, self.episodic_store,
            self.policy_store, self.knowledge_graph,
        )
        self.runtime = CognitiveRuntime(llm, self.policy_store, self.procedural_store)
        self.evolution = EvolutionSandbox(session, llm, audit_log=None)  # audit_log wired below
        self.audit_log = AuditLog(session)

        # GAP-M8: Wire audit log into evolution sandbox
        self.evolution._audit_log = self.audit_log

        # Wire evolution sandbox into compiler and retrieval after both are created
        self.compiler._evolution = self.evolution
        self.retrieval._evolution = self.evolution

        # GAP-5: StyleTracker is optional — only active if the domain schema
        # defines a 'communication_style' entity type, or if not checking.
        self.style_tracker: CommunicationStyleTracker | None = None
        try:
            from percos.schema import get_domain_schema
            schema = get_domain_schema()
            if "communication_style" in {t.lower() for t in schema.get_entity_type_names()}:
                self.style_tracker = CommunicationStyleTracker()
        except Exception:
            pass  # Schema not loaded yet; style tracker remains disabled

        self.consistency_checker = CrossMemoryChecker(session)
        self.deadline_checker = DeadlineChecker(self.semantic_store)
        self.evaluation = get_evaluation_harness()

        # GAP-L1: External integration manager
        self.integration_manager = IntegrationManager(self)

        # Working memory (session-scoped)
        self.working_memory = WorkingMemory()
        self._kg_loaded = False

    # ── Metrics access (§13) ────────────────────────────

    async def get_metrics(self) -> dict[str, Any]:
        """Return aggregated evaluation metrics (§13)."""
        return self.evaluation.get_summary()

    async def load_metrics(self) -> None:
        """Reload persisted metrics from DB on startup."""
        await self.evaluation.load_from_db(self.session)

    # ── Working memory persistence (Gap #18) ────────────

    async def save_working_memory(self) -> None:
        """Persist current working memory to DB."""
        state = self.working_memory.model_dump(mode="json")
        stmt = select(WorkingMemoryRow).where(WorkingMemoryRow.id == "default")
        result = await self.session.execute(stmt)
        row = result.scalar_one_or_none()
        if row:
            row.state = state
            from datetime import datetime, timezone
            row.updated_at = datetime.now(tz=timezone.utc)
        else:
            row = WorkingMemoryRow(id="default", state=state)
            self.session.add(row)
        await self.session.flush()

    async def load_working_memory(self) -> None:
        """Restore working memory from DB if available."""
        stmt = select(WorkingMemoryRow).where(WorkingMemoryRow.id == "default")
        result = await self.session.execute(stmt)
        row = result.scalar_one_or_none()
        if row and row.state:
            try:
                self.working_memory = WorkingMemory(**row.state)
                # Restore style tracker from session context (Gap #11)
                style_data = self.working_memory.session_context.get("communication_style")
                if style_data and self.style_tracker:
                    self.style_tracker.load_profile(style_data)
            except Exception:
                log.warning("working_memory_restore_failed")

    # ── Knowledge Graph sync ───────────────────────────

    async def _ensure_kg_loaded(self) -> None:
        """Load knowledge graph from DB (once per session)."""
        if self._kg_loaded:
            return
        facts = await self.semantic_store.get_active_facts()
        stmt = select(RelationRow)
        result = await self.session.execute(stmt)
        relations = list(result.scalars().all())
        self.knowledge_graph.load_from_facts_and_relations(facts, relations)
        self._kg_loaded = True
        log.info("kg_loaded", nodes=self.knowledge_graph.node_count, edges=self.knowledge_graph.edge_count)

    def _sync_kg_fact(self, fact_id: str, entity_type: str, entity_data: dict) -> None:
        """Add / update a single node in the KG after commit."""
        self.knowledge_graph.add_node(fact_id, entity_type=entity_type, data=entity_data)

    def _sync_kg_relation(self, src: str, tgt: str, **attrs) -> None:
        """Add an edge to the KG."""
        self.knowledge_graph.add_edge(src, tgt, **attrs)

    # ── Core API Surface (§16) ──────────────────────────

    async def ingest_event(self, event: RawEvent) -> str:
        """ingest_event(event) -> event_id"""
        # Security: sanitize and rate-limit (Gap #16)
        from percos.engine.security import sanitize_input, get_rate_limiter
        _, warnings = sanitize_input(event.content)
        if warnings:
            event.metadata_extra["injection_warnings"] = warnings
            log.warning("injection_flagged", source=event.source, warnings=warnings)

        limiter = get_rate_limiter()
        if not limiter.check(event.source or "anonymous"):
            log.warning("rate_limited_ingest", source=event.source)
            await self.audit_log.record(
                "event_ingested", "ingestion", actor=event.source or "anonymous",
                outcome="blocked", details={"reason": "rate_limited"},
            )
            raise ValueError(f"Rate limit exceeded for source: {event.source}")

        event_id = await self.ingestion.ingest(event)
        await self.audit_log.record(
            "event_ingested", "ingestion", actor=event.source or "anonymous",
            resource_id=event_id, resource_type="event",
            details={"event_type": str(event.event_type), "warnings": warnings},
        )
        return event_id

    async def compile_memory(self, event_id: str) -> list[dict]:
        """compile_memory(event_id) -> candidate_facts[]"""
        self.evaluation.start_timer("compile_memory")
        await self._ensure_kg_loaded()
        candidates = await self.compiler.compile(event_id)
        # Auto-commit accepted candidates
        committed = await self.compiler.auto_commit_accepted(candidates)
        # Sync committed facts and relations to KG (§7.4 – GAP-C2 / GAP-M6)
        for c in candidates:
            if str(c.id) in committed:
                self._sync_kg_fact(str(c.id), c.entity_type, c.entity_data)
                # Sync any relations that were committed for this candidate
                rels = await self.compiler.get_committed_relations(str(c.id))
                for rel in rels:
                    self._sync_kg_relation(
                        rel["source_id"], rel["target_id"],
                        relation_type=rel["relation_type"],
                    )

        # §13 — record compilation metrics
        from percos.models.enums import CandidateRouting
        needs_review = sum(
            1 for c in candidates
            if c.routing in (CandidateRouting.NEEDS_VERIFICATION, CandidateRouting.NEEDS_USER_CONFIRM)
        )
        self.evaluation.record_compilation(
            candidates_extracted=len(candidates),
            auto_committed=len(committed),
            needs_review=needs_review,
        )
        self.evaluation.stop_timer("compile_memory")

        return [
            {
                "candidate_id": str(c.id),
                "entity_type": c.entity_type,
                "routing": c.routing.value,
                "conflicts": [str(x) for x in c.conflicts_with],
                "committed": str(c.id) in committed,
            }
            for c in candidates
        ]

    async def validate_candidate(self, candidate_id: str, accept: bool = True) -> dict:
        """validate_candidate(candidate) -> {accept|confirm|quarantine}"""
        fact_id = await self.compiler.validate_and_commit(candidate_id, accept)

        # Sync any relations committed during validation to KG (§7.4 – GAP-C2)
        if fact_id:
            rels = await self.compiler.get_committed_relations(candidate_id)
            for rel in rels:
                self._sync_kg_relation(
                    rel["source_id"], rel["target_id"],
                    relation_type=rel["relation_type"],
                )

        await self.audit_log.record(
            "candidate_validated", "compiler",
            resource_id=candidate_id, resource_type="candidate",
            details={"accepted": accept, "fact_id": fact_id},
        )
        return {"accepted": accept, "fact_id": fact_id}

    async def query_world_model(self, query: str, task_type: str | None = None) -> ContextBundle:
        """query_world_model(query, task_type) -> context_bundle"""
        self.evaluation.start_timer("retrieval")
        await self._ensure_kg_loaded()
        bundle = await self.retrieval.retrieve(query, self.working_memory, task_type=task_type)
        self.evaluation.stop_timer("retrieval")

        # §13 — record retrieval metrics
        self.evaluation.record_retrieval(
            query=query,
            semantic_count=len(bundle.semantic_facts),
            episodic_count=len(bundle.episodic_entries),
            graph_entities=sum(
                len(v.get("nodes", [])) for v in bundle.graph_context.values()
            ) if bundle.graph_context else 0,
        )
        return bundle

    async def plan_action(self, query: str, context: ContextBundle) -> dict:
        """plan_action(context_bundle, policy) -> action_plan"""
        return await self.runtime.plan(query, context)

    async def execute_action(self, plan: dict, context: ContextBundle) -> dict:
        """execute_action(action_plan) -> outcome"""
        return await self.runtime.execute(plan, context)

    async def reflect_on_outcome(self, outcome: dict) -> dict:
        """reflect_on_outcome(outcome) -> lessons/proposals"""
        return await self.runtime.reflect(outcome)

    async def propose_evolution(self, change: dict) -> str:
        """propose_evolution(change) -> proposal_id"""
        return await self.evolution.propose(
            change_type=change.get("change_type", ""),
            description=change.get("description", ""),
            payload=change.get("payload", {}),
        )

    async def simulate_proposal(self, proposal_id: str, replay_set: list[dict] | None = None) -> dict:
        """simulate_proposal(proposal_id, replay_set) -> eval_report"""
        return await self.evolution.simulate(proposal_id, replay_set)

    async def promote_proposal(self, proposal_id: str) -> dict:
        """promote_proposal(proposal_id) -> deployment_version"""
        return await self.evolution.deploy(proposal_id)

    async def explain_belief(self, fact_id: str) -> dict:
        """explain_belief(fact_id) -> provenance_chain"""
        return await self.semantic_store.explain(fact_id)

    # ── Document Import (GAP-H10) ──────────────────────

    async def import_document(
        self,
        content: str,
        source: str = "document_import",
        title: str | None = None,
        content_type: str = "text/plain",
    ) -> dict[str, Any]:
        """Import a long-form document: chunk → ingest each chunk → compile.

        Returns dict with ``event_ids``, ``chunks``, and ``title``.
        """
        event_ids = await self.ingestion.import_document(
            content=content, source=source, title=title, content_type=content_type,
        )
        # Auto-compile each ingested chunk so facts are extracted immediately
        for eid in event_ids:
            try:
                await self.compile_memory(eid)
            except Exception:
                log.warning("compile failed for chunk event %s", eid)
        return {"event_ids": event_ids, "chunks": len(event_ids), "title": title}

    # ── High-level conversation interface ───────────────

    async def chat(self, user_message: str, source: str = "user_chat") -> dict[str, Any]:
        """Full cognitive pipeline: ingest → compile → retrieve → think → respond."""
        self.evaluation.start_timer("chat")

        # 0. Learn communication style (§14.3 – Gap #11)
        if self.style_tracker:
            style_features = self.style_tracker.analyse(user_message)
            self.working_memory.session_context["communication_style"] = self.style_tracker.get_profile()

        # GAP-H6: Extract goals and open questions from user message
        await self._update_working_memory_from_message(user_message)

        # 1. Ingest
        event = RawEvent(
            event_type="conversation",
            source=source,
            content=user_message,
        )
        event_id = await self.ingest_event(event)

        # 2. Compile memory
        candidates = await self.compile_memory(event_id)

        # 3. Retrieve context
        context = await self.query_world_model(user_message)

        # 4. Think and act
        result = await self.runtime.think_and_act(user_message, context)

        # GAP-H6: Update working memory with plan steps from the result
        await self._update_working_memory_from_result(result)

        # 5. Feed reflection results back into memory (Gap #3)
        reflection = result.get("reflection") or {}
        await self._apply_reflection(reflection)

        # GAP-H9: Periodically commit style profile as semantic fact
        await self._maybe_commit_style_profile()

        # 6. §13 — record agent-quality metrics from think_and_act result
        outcome = result.get("outcome")
        if outcome is not None:
            success = outcome.get("success", False) if isinstance(outcome, dict) else False
            self.evaluation.record_plan_outcome(success)
            self.evaluation.record_safe_action()

            # GAP-H11: Update procedural success rate based on outcome
            await self._update_procedural_success_rate(result, success)

        guardrail = result.get("guardrail_result")
        if guardrail is not None:
            self.evaluation.record_policy_compliance(guardrail.get("approved", True))
            if not guardrail.get("approved", True):
                self.evaluation.record_unsafe_action()

        # 7. Persist working memory (Gap #18)
        await self.save_working_memory()

        # 8. Persist metrics to DB periodically (§13)
        await self.evaluation.save_to_db(self.session)

        self.evaluation.stop_timer("chat")
        await self.session.commit()

        return {
            "event_id": event_id,
            "response": result.get("response", ""),
            "candidates_extracted": len(candidates),
            "plan": result.get("plan"),
            "reflection": reflection,
        }

    async def _update_working_memory_from_message(self, message: str) -> None:
        """GAP-H6: Extract goals and open questions from user message.

        Uses LLM to identify:
        - Active goals expressed in the message
        - Open questions that need answers
        - Any answered questions that can be cleared
        """
        try:
            prompt = (
                f"Analyze this user message and extract any goals or questions:\n\n"
                f"Message: {message}\n\n"
                f"Current active goals: {_json.dumps([str(g) for g in self.working_memory.active_goals], default=str)}\n"
                f"Current open questions: {_json.dumps(self.working_memory.open_questions, default=str)}\n\n"
                f"Respond with JSON:\n"
                f'{{"new_goals": ["goal1"], "answered_questions": ["q1"], '
                f'"new_questions": ["q1"], "is_goal_completion": false}}'
            )
            result = await self.llm.extract_structured(
                "You are a goal and question extraction engine for a personal assistant.", prompt
            )
            if isinstance(result, dict):
                # Add new goals
                for goal in result.get("new_goals", []):
                    if goal and goal not in [str(g) for g in self.working_memory.active_goals]:
                        self.working_memory.active_goals.append(goal)

                # Remove answered questions
                answered = set(result.get("answered_questions", []))
                if answered:
                    self.working_memory.open_questions = [
                        q for q in self.working_memory.open_questions
                        if q not in answered
                    ]

                # Add new questions
                for q in result.get("new_questions", []):
                    if q and q not in self.working_memory.open_questions:
                        self.working_memory.open_questions.append(q)

        except Exception:
            log.debug("working_memory_extraction_failed")

    async def _update_working_memory_from_result(self, result: dict) -> None:
        """GAP-H6: Update working memory with plan steps and clear completed goals."""
        plan = result.get("plan")
        if isinstance(plan, dict):
            steps = plan.get("steps", [])
            if steps:
                self.working_memory.current_plan = [
                    step.get("action", str(step)) if isinstance(step, dict) else str(step)
                    for step in steps
                ]
            # Track goal from plan
            goal = plan.get("goal")
            if goal and goal not in [str(g) for g in self.working_memory.active_goals]:
                self.working_memory.active_goals.append(goal)

        # If outcome was successful, consider the plan goal completed
        outcome = result.get("outcome")
        if isinstance(outcome, dict) and outcome.get("success"):
            # Clear the current plan as it's completed
            self.working_memory.current_plan = []

    async def _apply_reflection(self, reflection: dict) -> None:
        """Write reflection lessons back into memory and feed improvements to evolution."""
        # Store lessons as episodic entries
        lessons = reflection.get("lessons", [])
        if lessons:
            from percos.models.events import EpisodicEntry
            from percos.models.enums import MemoryType
            lesson_text = "Reflection lessons: " + "; ".join(lessons)
            entry = EpisodicEntry(
                event_id=UUID("00000000-0000-0000-0000-000000000000"),
                memory_type=MemoryType.META,
                content=lesson_text,
            )
            await self.episodic_store.append(entry)

        # Convert memory_updates from reflection into committed facts
        memory_updates = reflection.get("memory_updates", [])
        for mu in memory_updates:
            if not isinstance(mu, dict):
                continue
            mu_type = mu.get("type", "observation")
            confidence_map = {"high": "high", "medium": "medium", "low": "low"}
            conf = confidence_map.get(mu.get("confidence", "medium"), "medium")
            fact = CommittedFact(
                candidate_id=UUID("00000000-0000-0000-0000-000000000000"),
                entity_type=mu_type,
                entity_data={"description": mu.get("description", "")},
                fact_type=FactType.DERIVED,
                confidence=Confidence(conf),
                scope="global",
                sensitivity=Sensitivity.INTERNAL,
                source="reflection",
                provenance_chain=["reflection_feedback"],
            )
            fact_id = await self.semantic_store.commit(fact)
            self._sync_kg_fact(fact_id, mu_type, fact.entity_data)

        # Feed improvement proposals to evolution sandbox
        # GAP-M1: Optionally auto-run the evolution pipeline on reflection proposals
        from percos.config import get_settings
        auto_evolve = get_settings().auto_evolve_reflection
        proposals = reflection.get("improvement_proposals", [])
        for prop in proposals:
            if not isinstance(prop, dict):
                continue
            change_type = prop.get("type", "extraction_prompt")
            try:
                proposal_id = await self.evolution.propose(
                    change_type=change_type,
                    description=prop.get("description", ""),
                    payload={"source": "reflection", "details": prop},
                )
                # GAP-M1: Auto-validate and, for low-risk changes, run full pipeline
                if auto_evolve:
                    validation = await self.evolution.validate(proposal_id)
                    if isinstance(validation, dict) and validation.get("valid"):
                        is_low_risk = change_type in (
                            "extraction_prompt", "retrieval_heuristic", "ranking_strategy",
                        )
                        if is_low_risk:
                            # Low risk: run full pipeline (simulate → score → approve → deploy)
                            sim = await self.evolution.simulate(proposal_id)
                            scoring = await self.evolution.score(proposal_id, sim)
                            approval = await self.evolution.approve(proposal_id)
                            if isinstance(approval, dict) and approval.get("approved"):
                                await self.evolution.deploy(proposal_id)
                                log.info("reflection_auto_deployed",
                                         proposal_id=proposal_id, change_type=change_type)
                        else:
                            log.info("reflection_validated_needs_approval",
                                     proposal_id=proposal_id, change_type=change_type)
            except Exception:
                log.warning("reflection_proposal_failed", proposal=prop)

    # ── GAP-H9: Style Profile as Semantic Fact ──────────

    _STYLE_COMMIT_THRESHOLD = 10  # Commit after this many messages analysed

    async def _maybe_commit_style_profile(self) -> None:
        """GAP-H9: Periodically commit the style profile as a semantic fact.

        Only commits after sufficient messages have been analyzed (threshold).
        Updates the existing style fact if one already exists.
        Only active when style_tracker is enabled (GAP-5).
        """
        if not self.style_tracker:
            return
        profile = self.style_tracker.get_profile()
        n = profile.get("messages_analysed", 0)
        if n < self._STYLE_COMMIT_THRESHOLD:
            return

        # Only commit every N messages to avoid excessive writes
        if n % self._STYLE_COMMIT_THRESHOLD != 0:
            return

        # Check if a style fact already exists
        existing = await self.semantic_store.get_active_facts(entity_type="communication_style")
        if existing:
            # Update the existing fact's entity_data
            row = existing[0]
            row.entity_data = {
                "name": "user_communication_style",
                "profile": profile,
            }
            from datetime import datetime, timezone as _tz
            row.last_verified = datetime.now(tz=_tz.utc)
            await self.session.flush()
            log.info("style_profile_updated", messages=n, fact_id=row.id)
        else:
            # Commit as a new semantic fact
            fact = CommittedFact(
                candidate_id=UUID("00000000-0000-0000-0000-000000000000"),
                entity_type="communication_style",
                entity_data={
                    "name": "user_communication_style",
                    "profile": profile,
                },
                fact_type=FactType.DERIVED,
                confidence=Confidence.MEDIUM,
                scope="global",
                sensitivity=Sensitivity.PRIVATE,
                source="style_tracker",
                provenance_chain=["style_tracker_auto_commit"],
            )
            fact_id = await self.semantic_store.commit(fact)
            self._sync_kg_fact(fact_id, "communication_style", fact.entity_data)
            log.info("style_profile_committed", messages=n, fact_id=fact_id)

    # ── GAP-H11: Procedural Success Rate Update ────────

    async def _update_procedural_success_rate(self, result: dict, success: bool) -> None:
        """GAP-H11: Update procedural memory success_rate based on execution outcome.

        If the plan used a named procedure, update its success_rate.
        """
        plan = result.get("plan")
        if not isinstance(plan, dict):
            return

        for step in plan.get("steps", []):
            if not isinstance(step, dict):
                continue
            # Check if the step references a procedure by name
            action = step.get("action", "")
            tool = step.get("tool", "")
            reference = action or tool
            if not reference:
                continue

            # Try to find matching procedures
            try:
                matching = await self.procedural_store.find_by_trigger(reference)
                for proc in matching:
                    # Update success rate with exponential moving average
                    old_rate = proc.success_rate or 0.0
                    alpha = 0.3  # learning rate
                    new_rate = old_rate * (1 - alpha) + (1.0 if success else 0.0) * alpha
                    await self.procedural_store.update(
                        proc.id, {"success_rate": round(new_rate, 4)}
                    )
                    log.info("procedural_success_rate_updated",
                             procedure_id=proc.id, old_rate=old_rate, new_rate=new_rate)
            except Exception:
                log.debug("procedural_success_rate_update_failed", reference=reference)

    # ── Memory inspection (§12) ─────────────────────────

    async def get_beliefs(self, entity_type: str | None = None) -> list[dict]:
        """Get all active beliefs, optionally filtered by entity type."""
        facts = await self.semantic_store.get_active_facts(entity_type)
        return [
            {
                "fact_id": f.id,
                "entity_type": f.entity_type,
                "entity_data": f.entity_data,
                "confidence": f.confidence,
                "scope": f.scope,
                "sensitivity": f.sensitivity,
                "source": f.source,
                "belief_status": f.belief_status,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in facts
        ]

    async def get_belief_history(self, entity_name: str) -> list[dict]:
        """Get the history of beliefs about an entity."""
        return await self.ttm.get_belief_history(entity_name)

    async def delete_belief(self, fact_id: str) -> dict:
        """Allow user to retract a belief."""
        await self.semantic_store.retract(fact_id)
        await self.audit_log.record(
            "belief_deleted", "brain", actor="user",
            resource_id=fact_id, resource_type="fact",
        )
        await self.session.commit()
        return {"retracted": True, "fact_id": fact_id}

    async def update_belief(self, fact_id: str, updates: dict) -> dict:
        """Allow user to edit a belief.

        Also refreshes ``last_verified`` so the fact is no longer considered
        stale after a user-confirmed edit (§6.3 / GAP-C3).
        """
        from datetime import datetime, timezone as _tz
        row = await self.semantic_store.get(fact_id)
        if not row:
            return {"error": "fact_not_found"}
        if "entity_data" in updates:
            row.entity_data = {**row.entity_data, **updates["entity_data"]}
        if "confidence" in updates:
            row.confidence = updates["confidence"]
        if "scope" in updates:
            row.scope = updates["scope"]
        if "sensitivity" in updates:
            row.sensitivity = updates["sensitivity"]
        # Re-verification timestamp (§6.3)
        row.last_verified = datetime.now(tz=_tz.utc)
        await self.audit_log.record(
            "belief_updated", "brain", actor="user",
            resource_id=fact_id, resource_type="fact",
            details={"fields_updated": list(updates.keys())},
        )
        await self.session.commit()
        return {"updated": True, "fact_id": fact_id}

    async def run_maintenance(self) -> dict:
        """Run periodic maintenance: staleness detection, contradiction scan, cross-memory consistency.

        GAP-H1: Also creates re-verification candidates for stale facts and
        uses LLM to propose updated hypotheses based on recent episodic evidence.
        """
        self.evaluation.start_timer("maintenance")

        stale = await self.ttm.detect_stale_facts()
        stale_ids = [f.id for f in stale]
        marked = await self.ttm.mark_stale(stale_ids)

        # GAP-H1: Create re-verification candidates for stale facts
        reverify_candidates = []
        for fact_row in stale:
            candidate = await self._create_reverification_candidate(fact_row)
            if candidate:
                reverify_candidates.append(candidate)

        # Contradiction scanning (§8.2 – GAP-C5)
        contradictions = await self.ttm.scan_contradictions()

        # Cross-memory consistency checks (§11.2 – Gap #12)
        consistency_issues = await self.consistency_checker.run()

        # §13 — record memory health metrics
        active_facts = await self.semantic_store.get_active_facts()
        self.evaluation.record_memory_stats(
            active_facts=len(active_facts),
            stale_facts=len(stale),
            contradictions=len(contradictions),
        )

        await self.audit_log.record(
            "maintenance_run", "ttm",
            details={
                "stale_detected": len(stale),
                "marked_stale": marked,
                "contradictions_found": len(contradictions),
                "consistency_issues": len(consistency_issues),
                "reverification_candidates": len(reverify_candidates),
            },
        )

        # Persist metrics
        await self.evaluation.save_to_db(self.session)

        await self.session.commit()
        # Reload KG after maintenance
        self._kg_loaded = False
        await self._ensure_kg_loaded()

        self.evaluation.stop_timer("maintenance")
        return {
            "stale_detected": len(stale),
            "marked_stale": marked,
            "contradictions": contradictions,
            "consistency_issues": consistency_issues,
            "reverification_candidates": len(reverify_candidates),
        }

    async def _create_reverification_candidate(self, fact_row) -> dict | None:
        """GAP-H1: Create a pending candidate for a stale fact so the user can re-verify it.

        Optionally uses LLM to propose an updated hypothesis based on recent
        episodic evidence.
        """
        from percos.models.enums import CandidateRouting

        # Gather recent episodic evidence related to this entity
        entity_name = (fact_row.entity_data or {}).get("name", "")
        hypothesis = None
        if entity_name:
            try:
                recent_episodes = await self.episodic_store.search(entity_name, limit=5)
                if recent_episodes:
                    episode_texts = "\n".join(
                        f"- {e.content[:200]}" for e in recent_episodes[:5]
                    )
                    prompt = (
                        f"A previously known fact is now stale and needs re-verification.\n\n"
                        f"Original fact ({fact_row.entity_type}): {_json.dumps(fact_row.entity_data, default=str)}\n\n"
                        f"Recent relevant episodes:\n{episode_texts}\n\n"
                        f"Based on the recent evidence, propose an updated hypothesis for this fact. "
                        f"Respond with JSON: {{\"hypothesis\": \"...\", \"confidence\": \"high|medium|low\", "
                        f"\"reasoning\": \"...\"}}"
                    )
                    hypothesis = await self.llm.extract_structured(
                        "You are a fact re-verification engine.", prompt
                    )
            except Exception:
                log.warning("reverification_hypothesis_failed", fact_id=fact_row.id)

        # Create a candidate fact row for user review
        candidate_row = CandidateFactRow(
            event_id=fact_row.candidate_id or "00000000-0000-0000-0000-000000000000",
            entity_type=fact_row.entity_type,
            entity_data={
                **(fact_row.entity_data or {}),
                "_stale_fact_id": fact_row.id,
                "_reverification": True,
                "_hypothesis": hypothesis if isinstance(hypothesis, dict) else None,
            },
            fact_type=fact_row.fact_type,
            confidence=fact_row.confidence,
            scope=fact_row.scope,
            sensitivity=fact_row.sensitivity,
            routing=CandidateRouting.NEEDS_USER_CONFIRM.value,
            provenance=[f"reverification:{fact_row.id}"],
            conflicts_with=[],
            raw_extraction=_json.dumps({
                "source": "stale_reverification",
                "original_fact_id": fact_row.id,
            }),
        )
        self.session.add(candidate_row)
        await self.session.flush()
        log.info("reverification_candidate_created",
                 fact_id=fact_row.id, candidate_id=candidate_row.id)
        return {
            "candidate_id": candidate_row.id,
            "original_fact_id": fact_row.id,
            "entity_type": fact_row.entity_type,
            "hypothesis": hypothesis if isinstance(hypothesis, dict) else None,
        }

    # ── Direct fact commit (Gap #6) ─────────────────────

    async def commit_fact(self, data: dict) -> dict:
        """Directly commit a user-provided fact into semantic memory."""
        from datetime import datetime, timezone as _tz
        fact = CommittedFact(
            candidate_id=UUID("00000000-0000-0000-0000-000000000000"),
            entity_type=data.get("entity_type", "observation"),
            entity_data=data.get("entity_data", {}),
            fact_type=FactType(data.get("fact_type", "observed")),
            confidence=Confidence(data.get("confidence", "high")),
            scope=data.get("scope", "global"),
            sensitivity=Sensitivity(data.get("sensitivity", "internal")),
            source=data.get("source", "user_direct"),
            provenance_chain=["user_direct_commit"],
            valid_from=data["valid_from"] if data.get("valid_from") else datetime.now(tz=_tz.utc),
            valid_to=data.get("valid_to"),
        )
        fact_id = await self.semantic_store.commit(fact)
        self._sync_kg_fact(fact_id, fact.entity_type, fact.entity_data)
        await self.audit_log.record(
            "fact_committed", "brain", actor=data.get("source", "user_direct"),
            resource_id=fact_id, resource_type="fact",
            details={"entity_type": fact.entity_type, "confidence": data.get("confidence", "high")},
        )
        await self.session.commit()
        return {"fact_id": fact_id, "committed": True}

    # ── Pending candidates (Gap #7) ─────────────────────

    async def list_pending_candidates(self) -> list[dict]:
        """List candidates that need user review (including quarantined — GAP-H2)."""
        from percos.models.enums import CandidateRouting
        stmt = select(CandidateFactRow).where(
            CandidateFactRow.routing.in_([
                CandidateRouting.NEEDS_VERIFICATION.value,
                CandidateRouting.NEEDS_USER_CONFIRM.value,
                CandidateRouting.QUARANTINE.value,
            ])
        )
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        return [
            {
                "candidate_id": r.id,
                "entity_type": r.entity_type,
                "entity_data": r.entity_data,
                "confidence": r.confidence,
                "routing": r.routing,
                "conflicts_with": r.conflicts_with,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    async def list_quarantined_candidates(self) -> list[dict]:
        """List only quarantined candidates (GAP-H2)."""
        from percos.models.enums import CandidateRouting
        stmt = select(CandidateFactRow).where(
            CandidateFactRow.routing == CandidateRouting.QUARANTINE.value
        )
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        return [
            {
                "candidate_id": r.id,
                "entity_type": r.entity_type,
                "entity_data": r.entity_data,
                "confidence": r.confidence,
                "routing": r.routing,
                "conflicts_with": r.conflicts_with,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    # ── Export / Reset (Gap #8) ─────────────────────────

    async def export_memory(self) -> dict:
        """Export all memories as JSON-serialisable dict."""
        facts = await self.semantic_store.get_active_facts()
        episodes = await self.episodic_store.list_recent(limit=1000)
        from percos.stores.procedural_policy_stores import ProceduralStore
        procedures = await self.procedural_store.list_all()
        policies = await self.policy_store.get_active_policies()
        return {
            "semantic_facts": [
                {
                    "id": f.id, "entity_type": f.entity_type,
                    "entity_data": f.entity_data, "confidence": f.confidence,
                    "scope": f.scope, "belief_status": f.belief_status,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in facts
            ],
            "episodic_entries": [
                {
                    "id": e.id, "event_id": e.event_id,
                    "content": e.content,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                }
                for e in episodes
            ],
            "procedural_entries": [
                {"id": p.id, "name": p.name, "description": p.description}
                for p in procedures
            ],
            "policies": [
                {"id": p.id, "name": p.name, "rule": p.rule, "effect": p.effect}
                for p in policies
            ],
        }

    async def reset_memory(self, confirm: bool = False) -> dict:
        """Reset all memories. Requires confirm=True for safety."""
        if not confirm:
            return {"error": "Pass confirm=True to reset all memory."}
        from percos.stores.tables import (
            EventRow, CandidateFactRow, CommittedFactRow,
            EpisodicRow, ProceduralRow, ProceduralVersionRow,
            PolicyRow, RelationRow, ProposalRow, AuditLogRow,
        )
        # Record the reset action before wiping
        await self.audit_log.record(
            "memory_reset", "brain", actor="user",
            details={"confirmed": True},
        )
        for table_cls in [
            ProposalRow, RelationRow, PolicyRow, ProceduralVersionRow, ProceduralRow,
            EpisodicRow, CommittedFactRow, CandidateFactRow, EventRow, AuditLogRow,
        ]:
            await self.session.execute(
                select(table_cls).execution_options(synchronize_session="fetch")
            )
            from sqlalchemy import delete
            await self.session.execute(delete(table_cls))
        await self.session.commit()
        self.knowledge_graph = KnowledgeGraph()
        self._kg_loaded = True
        self.working_memory = WorkingMemory()
        return {"reset": True}

    # ── Generic Entity CRUD (Domain-Agnostic) ─────────

    async def create_entity(self, entity_type: str, data: dict) -> dict:
        """Create an entity of any type as a committed fact.

        This is the domain-agnostic replacement for create_task / create_project.
        The entity type is validated against the active domain schema.
        """
        # Validate entity_type against schema (GAP-4: §18 #2)
        try:
            from percos.schema import get_domain_schema
            schema = get_domain_schema()
            valid_types = {t.lower() for t in schema.get_entity_type_names()}
            if entity_type.lower() not in valid_types:
                log.warning("unknown_entity_type", entity_type=entity_type,
                            valid_types=list(valid_types))
                return {"error": f"Unknown entity type '{entity_type}'. "
                        f"Valid types: {sorted(valid_types)}",
                        "entity_type": entity_type, "committed": False}
        except Exception:
            log.debug("schema_validation_skipped")

        entity_data = dict(data)
        # Remove meta keys that go into the fact wrapper, not entity_data
        source = entity_data.pop("source", f"{entity_type}_management")
        fact_type = entity_data.pop("fact_type", "observed")
        confidence = entity_data.pop("confidence", "high")
        scope = entity_data.pop("scope", "global")
        sensitivity = entity_data.pop("sensitivity", "internal")

        result = await self.commit_fact({
            "entity_type": entity_type,
            "entity_data": entity_data,
            "source": source,
            "fact_type": fact_type,
            "confidence": confidence,
            "scope": scope,
            "sensitivity": sensitivity,
        })
        await self.audit_log.record(
            "entity_created", "brain", actor="user",
            resource_id=result["fact_id"], resource_type=entity_type,
            details={"name": entity_data.get("name", ""), "entity_type": entity_type},
        )
        return {"entity_id": result["fact_id"], "entity_type": entity_type, "committed": True}

    async def list_entities(
        self,
        entity_type: str,
        filters: dict | None = None,
    ) -> list[dict]:
        """List active facts of a given entity type with optional field filters.

        This is the domain-agnostic replacement for list_tasks / list_projects.
        """
        facts = await self.semantic_store.get_active_facts(entity_type=entity_type)
        entities: list[dict] = []
        for f in facts:
            data = f.entity_data or {}
            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if data.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            entities.append({
                "entity_id": f.id,
                "entity_type": entity_type,
                **data,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "confidence": f.confidence,
                "scope": f.scope,
            })
        return entities

    async def get_entity(self, entity_id: str) -> dict | None:
        """Get a single entity by its fact ID."""
        stmt = select(CommittedFactRow).where(
            CommittedFactRow.id == entity_id,
            CommittedFactRow.belief_status == BeliefStatus.ACTIVE,
        )
        result = await self.session.execute(stmt)
        row = result.scalar_one_or_none()
        if not row:
            return None
        return {
            "entity_id": row.id,
            "entity_type": row.entity_type,
            **(row.entity_data or {}),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "confidence": row.confidence,
            "scope": row.scope,
        }

    async def update_entity(self, entity_id: str, updates: dict) -> dict:
        """Update an entity's data fields. Alias for update_belief."""
        result = await self.update_belief(entity_id, {"entity_data": updates})
        self.evaluation.record_user_correction()
        return result

    async def delete_entity(self, entity_id: str) -> dict:
        """Retract an entity. Alias for delete_belief."""
        result = await self.delete_belief(entity_id)
        self.evaluation.record_user_correction()
        return result

    # ── Priority Suggestion / Next-Action (§14.3) ─

    async def suggest_next_actions(self) -> list[dict]:
        """Generate a prioritised list of suggested next actions.

        Uses schema-driven entity discovery: queries entities that have
        deadline/status fields as defined by the domain schema, plus
        plan steps from working memory (§14.3).
        """
        from datetime import datetime, timezone as _tz
        from percos.schema import get_domain_schema

        goals = self.working_memory.active_goals
        plan = self.working_memory.current_plan

        suggestions: list[dict] = []
        now = datetime.now(tz=_tz.utc)

        # 1. Active plan steps → immediate suggestions
        for i, step in enumerate(plan):
            suggestions.append({
                "type": "plan_step",
                "priority": 100 - i,
                "title": step if isinstance(step, str) else str(step),
                "reason": "Part of your current plan",
            })

        # 2. Deadline-checker reminders
        try:
            reminders = await self.deadline_checker.get_reminders(horizon_days=7)
            for r in reminders:
                urgency_priority = {"overdue": 95, "urgent": 85, "soon": 60, "upcoming": 30}
                suggestions.append({
                    "type": "deadline_reminder",
                    "priority": urgency_priority.get(r.get("urgency"), 20),
                    "title": r.get("message", r.get("name", "")),
                    "fact_id": r.get("fact_id"),
                    "deadline": r.get("deadline"),
                    "urgency": r.get("urgency"),
                    "reason": f"Deadline: {r.get('urgency', 'upcoming')}",
                })
        except Exception:
            log.debug("deadline_reminders_failed")

        # 3. Schema-driven: find entity types with a status or deadline field
        schema = get_domain_schema()
        actionable_types: list[str] = []
        for et in schema.entity_types.values():
            field_names = {f.name for f in et.fields}
            if "status" in field_names or "deadline" in field_names:
                actionable_types.append(et.name.lower())

        for etype in actionable_types:
            facts = await self.semantic_store.get_active_facts(entity_type=etype)
            for f in facts:
                data = f.entity_data or {}
                status = data.get("status", "")
                if status in ("done", "cancelled", "completed"):
                    continue
                deadline_str = data.get("deadline")
                priority_val = data.get("priority", 0)
                if isinstance(priority_val, str):
                    try:
                        priority_val = int(priority_val)
                    except (ValueError, TypeError):
                        priority_val = 0
                urgency = priority_val
                if deadline_str:
                    try:
                        dl = datetime.fromisoformat(str(deadline_str))
                        if dl.tzinfo is None:
                            dl = dl.replace(tzinfo=_tz.utc)
                        days_left = (dl - now).total_seconds() / 86400
                        if days_left < 0:
                            urgency += 50
                        elif days_left < 1:
                            urgency += 40
                        elif days_left < 3:
                            urgency += 30
                        elif days_left < 7:
                            urgency += 20
                        else:
                            urgency += 10
                    except (ValueError, TypeError):
                        pass
                urgency += priority_val * 5
                suggestions.append({
                    "type": etype,
                    "priority": urgency,
                    "title": data.get("name", f.entity_type),
                    "entity_id": str(f.id),
                    "status": status,
                    "deadline": deadline_str,
                    "reason": f"{etype.title()} needs attention",
                })

        # 4. Active goals → goal reminders
        for g in goals:
            suggestions.append({
                "type": "goal",
                "priority": 15,
                "title": g if isinstance(g, str) else str(g),
                "reason": "Active goal in working memory",
            })

        # Sort by priority descending
        suggestions.sort(key=lambda s: s.get("priority", 0), reverse=True)
        return suggestions
