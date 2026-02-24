"""Cognitive Runtime – Planner, Executor, Critic / Guardrail Checker (§4.B).

Handles:
- plan_action(context_bundle, policy) -> action_plan
- execute_action(action_plan) -> outcome
- reflect_on_outcome(outcome) -> lessons / proposals
- Tool registration and invocation framework
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable
from uuid import uuid4

from percos.llm import LLMClient
from percos.logging import get_logger
from percos.models.events import ContextBundle, PolicyEntry

log = get_logger("runtime")

# ── Tool Registry ──────────────────────────────────────

ToolFn = Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Registry of callable tools available to the cognitive runtime."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        fn: ToolFn,
        description: str = "",
        params_schema: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = {
            "fn": fn,
            "description": description,
            "params_schema": params_schema or {},
        }

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": k, "description": v["description"], "params_schema": v["params_schema"]}
            for k, v in self._tools.items()
        ]

    async def invoke(self, name: str, params: dict[str, Any] | None = None) -> Any:
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Tool '{name}' not found", "available": list(self._tools.keys())}
        try:
            result = await tool["fn"](**(params or {}))
            return result
        except Exception as exc:
            log.error("tool_invocation_failed", tool=name, error=str(exc))
            return {"error": str(exc), "traceback": traceback.format_exc()}

    def has(self, name: str) -> bool:
        return name in self._tools

PLANNER_SYSTEM_PROMPT = """\
You are the planning module of a Personal Cognitive OS.
Given the user's query and a context bundle (facts, episodic memories, policies, graph),
create a concrete action plan.

Respond with a JSON object:
{
  "plan_id": "<generated uuid>",
  "goal": "what the user wants",
  "steps": [
    {"step_id": 1, "action": "description", "tool": "optional_tool_name", "params": {}},
    ...
  ],
  "reasoning": "why this plan",
  "risks": ["potential risks"],
  "requires_approval": true/false
}
"""

EXECUTOR_SYSTEM_PROMPT = """\
You are the execution module of a Personal Cognitive OS.
Given an action plan and context, execute each step and produce the final response.
If a step requires a tool you don't have, describe what the tool would do.

Respond with a JSON object:
{
  "outcome_id": "<uuid>",
  "success": true/false,
  "result": "the result or response to give the user",
  "steps_completed": [{"step_id": 1, "status": "done", "output": "..."}],
  "side_effects": ["any memory updates or changes to note"]
}
"""

CRITIC_SYSTEM_PROMPT = """\
You are the critic / guardrail checker for a Personal Cognitive OS.
Given an action plan and the active policies, check if the plan is safe to execute.

Respond with JSON:
{
  "approved": true/false,
  "violations": [{"policy": "name", "reason": "why it violates"}],
  "warnings": ["non-blocking concerns"],
  "suggestions": ["improvements"]
}
"""

REFLECTION_SYSTEM_PROMPT = """\
You are the reflection module of a Personal Cognitive OS.
Given an action outcome, identify lessons learned, potential memory updates,
and improvement proposals.

Respond with JSON:
{
  "lessons": ["lesson1", "lesson2"],
  "memory_updates": [
    {"type": "preference|fact|skill", "description": "what to remember", "confidence": "high|medium|low"}
  ],
  "improvement_proposals": [
    {"type": "extraction|retrieval|skill", "description": "proposed improvement"}
  ]
}
"""


class CognitiveRuntime:
    """The 'thinking' core: plan, check, execute, reflect."""

    def __init__(self, llm: LLMClient, policy_store=None, procedural_store=None):
        self._llm = llm
        self._policy_store = policy_store
        self._procedural_store = procedural_store
        self.tools = ToolRegistry()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register built-in domain-agnostic tools."""

        async def tool_get_current_time(**_: Any) -> dict[str, str]:
            """Return the current UTC timestamp."""
            return {"utc": datetime.now(timezone.utc).isoformat() + "Z"}

        async def tool_calculate(expression: str = "", **_: Any) -> dict[str, Any]:
            """Safely evaluate a simple arithmetic expression."""
            allowed = set("0123456789+-*/.() ")
            if not expression or not all(c in allowed for c in expression):
                return {"error": "Only simple arithmetic expressions are allowed"}
            try:
                result = eval(expression, {"__builtins__": {}})  # noqa: S307
                return {"expression": expression, "result": result}
            except Exception as exc:
                return {"error": str(exc)}

        async def tool_summarise(text: str = "", max_length: int = 200, **_: Any) -> dict[str, str]:
            """Summarise a block of text using the LLM."""
            if not text:
                return {"error": "No text provided"}
            prompt = f"Summarise the following in at most {max_length} characters:\n\n{text}"
            summary = await self._llm.generate(prompt)
            return {"summary": summary}

        async def tool_list_entity_types(**_: Any) -> dict[str, Any]:
            """List available entity types from the domain schema."""
            try:
                from percos.schema import get_domain_schema
                schema = get_domain_schema()
                return {"entity_types": schema.get_entity_type_names()}
            except Exception as exc:
                return {"error": str(exc)}

        self.tools.register(
            "get_current_time", tool_get_current_time,
            description="Get the current UTC date and time",
            params_schema={},
        )
        self.tools.register(
            "calculate", tool_calculate,
            description="Evaluate a simple arithmetic expression",
            params_schema={"expression": {"type": "string", "description": "e.g. '2 + 3 * 4'"}},
        )
        self.tools.register(
            "summarise", tool_summarise,
            description="Summarise a block of text",
            params_schema={
                "text": {"type": "string", "description": "Text to summarise"},
                "max_length": {"type": "integer", "description": "Max output length (chars)"},
            },
        )
        self.tools.register(
            "list_entity_types", tool_list_entity_types,
            description="List available entity types from the active domain schema",
            params_schema={},
        )

    # ── Plan ────────────────────────────────────────────
    async def plan(self, query: str, context: ContextBundle) -> dict[str, Any]:
        """Generate an action plan from query + context.

        Core interface: plan_action(context_bundle, policy) -> action_plan

        GAP-H5: Queries procedural memory for matching skills/workflows
        and includes them in the planner context.
        """
        context_summary = self._summarize_context(context)
        tools_description = ""
        if self.tools.list_tools():
            tools_description = "\n\nAvailable tools:\n" + "\n".join(
                f"- {t['name']}: {t['description']}" for t in self.tools.list_tools()
            )

        # GAP-H5: Query procedural memory for relevant skills
        procedures_description = ""
        if self._procedural_store:
            try:
                matching_procs = await self._procedural_store.find_by_trigger(query)
                if matching_procs:
                    procedures_description = "\n\nAvailable learned procedures/skills:\n" + "\n".join(
                        f"- {p.name} (trigger: {p.trigger}, success_rate: {p.success_rate:.2f}, v{p.version}): "
                        f"{p.description}\n  Steps: {', '.join(p.steps[:5])}"
                        for p in matching_procs[:5]
                    )
                    procedures_description += (
                        "\n\nPrefer using these learned procedures when they match the task. "
                        "Reference them by name in your plan steps."
                    )
            except Exception:
                pass

        user_content = (
            f"User query: {query}\n\n"
            f"Context:\n{context_summary}"
            f"{tools_description}"
            f"{procedures_description}"
        )
        try:
            plan = await self._llm.extract_structured(PLANNER_SYSTEM_PROMPT, user_content)
            if not isinstance(plan, dict):
                plan = {"plan_id": str(uuid4()), "goal": query, "steps": [], "reasoning": "failed to plan"}
        except Exception as exc:
            log.error("planning_failed", error=str(exc))
            plan = {"plan_id": str(uuid4()), "goal": query, "steps": [], "error": str(exc)}

        log.info("plan_created", plan_id=plan.get("plan_id"), steps=len(plan.get("steps", [])))
        return plan

    # ── Critic / Guardrail ──────────────────────────────
    async def check_guardrails(
        self, plan: dict[str, Any], policies: list[PolicyEntry]
    ) -> dict[str, Any]:
        """Check if an action plan complies with active policies."""
        import json
        policy_text = "\n".join(
            f"- {p.name}: {p.rule} (effect: {p.effect})" for p in policies
        )
        user_content = (
            f"Action Plan:\n{json.dumps(plan, indent=2, default=str)}\n\n"
            f"Active Policies:\n{policy_text or '(no policies defined)'}"
        )
        try:
            result = await self._llm.extract_structured(CRITIC_SYSTEM_PROMPT, user_content)
            return result if isinstance(result, dict) else {"approved": True}
        except Exception as exc:
            log.error("guardrail_check_failed", error=str(exc))
            return {"approved": False, "error": str(exc)}

    # ── Execute ─────────────────────────────────────────
    async def execute(self, plan: dict[str, Any], context: ContextBundle) -> dict[str, Any]:
        """Execute an action plan, invoking real tools when available.

        Core interface: execute_action(action_plan) -> outcome
        """
        import json
        steps_completed: list[dict] = []
        side_effects: list[str] = []

        for step in plan.get("steps", []):
            tool_name = step.get("tool")
            params = step.get("params", {})
            step_id = step.get("step_id", len(steps_completed) + 1)

            if tool_name and self.tools.has(tool_name):
                # Real tool execution
                tool_result = await self.tools.invoke(tool_name, params)
                steps_completed.append({
                    "step_id": step_id,
                    "status": "done",
                    "tool": tool_name,
                    "output": tool_result if isinstance(tool_result, (str, dict, list)) else str(tool_result),
                })
                side_effects.append(f"Executed tool: {tool_name}")
            else:
                # LLM-simulated step
                steps_completed.append({
                    "step_id": step_id,
                    "status": "simulated",
                    "action": step.get("action", ""),
                    "output": f"(no tool '{tool_name}' registered)" if tool_name else "(LLM reasoning step)",
                })

        # Generate final response via LLM using step outputs
        context_summary = self._summarize_context(context)
        step_summaries = json.dumps(steps_completed, indent=2, default=str)
        user_content = (
            f"Action Plan:\n{json.dumps(plan, indent=2, default=str)}\n\n"
            f"Step Results:\n{step_summaries}\n\n"
            f"Context:\n{context_summary}"
        )
        try:
            outcome = await self._llm.extract_structured(EXECUTOR_SYSTEM_PROMPT, user_content)
            if not isinstance(outcome, dict):
                outcome = {"outcome_id": str(uuid4()), "success": False, "result": "execution failed"}
        except Exception as exc:
            log.error("execution_failed", error=str(exc))
            outcome = {"outcome_id": str(uuid4()), "success": False, "error": str(exc)}

        outcome.setdefault("outcome_id", str(uuid4()))
        outcome["steps_completed"] = steps_completed
        outcome["side_effects"] = side_effects

        log.info("execution_complete", outcome_id=outcome.get("outcome_id"), success=outcome.get("success"))
        return outcome

    # ── Reflect ─────────────────────────────────────────
    async def reflect(self, outcome: dict[str, Any]) -> dict[str, Any]:
        """Reflect on an outcome to extract lessons and proposals.

        Core interface: reflect_on_outcome(outcome) -> lessons/proposals
        """
        import json
        user_content = f"Action Outcome:\n{json.dumps(outcome, indent=2, default=str)}"
        try:
            reflection = await self._llm.extract_structured(REFLECTION_SYSTEM_PROMPT, user_content)
            return reflection if isinstance(reflection, dict) else {"lessons": [], "memory_updates": []}
        except Exception as exc:
            log.error("reflection_failed", error=str(exc))
            return {"lessons": [], "error": str(exc)}

    # ── Full pipeline ───────────────────────────────────
    async def think_and_act(self, query: str, context: ContextBundle) -> dict[str, Any]:
        """Full cognitive loop: plan → policy check → guardrail → execute → reflect."""
        from percos.engine.evaluation import get_evaluation_harness
        harness = get_evaluation_harness()
        harness.start_timer("think_and_act")

        # 1. Plan
        plan = await self.plan(query, context)

        # 2. Pre-action policy enforcement (Gap #15)
        if self._policy_store:
            for step in plan.get("steps", []):
                action = step.get("action", "") or step.get("tool", "")
                if action:
                    policy_check = await self._policy_store.check_action(action)
                    if policy_check and not policy_check.get("allowed", True):
                        harness.record_policy_compliance(False)
                        harness.stop_timer("think_and_act")
                        return {
                            "response": f"Action blocked by policy '{policy_check.get('policy_name', 'unknown')}': {policy_check.get('reason', '')}",
                            "plan": plan,
                            "guardrail_result": {"approved": False, "policy_violation": policy_check},
                            "outcome": None,
                            "reflection": None,
                        }

        # 3. Check guardrails
        guardrail_result = await self.check_guardrails(plan, context.policies)
        if not guardrail_result.get("approved", True):
            harness.record_policy_compliance(False)
            harness.record_unsafe_action()
            harness.stop_timer("think_and_act")
            return {
                "response": "Action blocked by safety policies.",
                "plan": plan,
                "guardrail_result": guardrail_result,
                "outcome": None,
                "reflection": None,
            }

        # Guardrails approved
        harness.record_policy_compliance(True)

        # 4. Execute
        outcome = await self.execute(plan, context)
        success = outcome.get("success", False) if isinstance(outcome, dict) else False
        harness.record_plan_outcome(success)
        if success:
            harness.record_safe_action()
        else:
            harness.record_unsafe_action()

        # 5. Reflect
        reflection = await self.reflect(outcome)

        harness.stop_timer("think_and_act")
        return {
            "response": outcome.get("result", ""),
            "plan": plan,
            "guardrail_result": guardrail_result,
            "outcome": outcome,
            "reflection": reflection,
        }

    def _summarize_context(self, context: ContextBundle) -> str:
        """Create a text summary of the context bundle for LLM consumption."""
        parts = []

        if context.semantic_facts:
            parts.append("### Known Facts")
            for f in context.semantic_facts[:20]:
                parts.append(f"- [{f.entity_type}] {f.entity_data} (confidence: {f.confidence})")

        if context.episodic_entries:
            parts.append("\n### Recent Episodes")
            for e in context.episodic_entries[:10]:
                parts.append(f"- {e.content[:200]}")

        if context.policies:
            parts.append("\n### Active Policies")
            for p in context.policies:
                parts.append(f"- {p.name}: {p.rule} ({p.effect})")

        if context.graph_context:
            parts.append("\n### Knowledge Graph Context")
            for entity, subgraph in context.graph_context.items():
                nodes_count = len(subgraph.get("nodes", []))
                edges_count = len(subgraph.get("edges", []))
                parts.append(f"- {entity}: {nodes_count} nodes, {edges_count} edges")

        if context.working_memory.active_goals:
            parts.append("\n### Active Goals")
            for g in context.working_memory.active_goals:
                parts.append(f"- {g}")

        return "\n".join(parts) if parts else "(no context available)"
