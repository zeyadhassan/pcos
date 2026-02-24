"""Shared enumerations used across the entire system."""

from __future__ import annotations

from enum import Enum


# ── Fact / Belief Classification (§7.1) ──────────────────
class FactType(str, Enum):
    """Every memory item must be classified as one of these."""
    OBSERVED = "observed"       # direct user statement or trusted tool output
    DERIVED = "derived"         # inferred from data
    HYPOTHESIS = "hypothesis"   # uncertain candidate belief
    POLICY = "policy"           # behavioral rule / permission


# ── Confidence ───────────────────────────────────────────
class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Sensitivity ──────────────────────────────────────────
class Sensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"
    SECRET = "secret"


# ── Candidate Routing (§8.4) ────────────────────────────
class CandidateRouting(str, Enum):
    AUTO_ACCEPT = "auto_accept"
    NEEDS_VERIFICATION = "needs_verification"
    NEEDS_USER_CONFIRM = "needs_user_confirm"
    QUARANTINE = "quarantine"


# ── Event Types ──────────────────────────────────────────
class EventType(str, Enum):
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    TOOL_RESULT = "tool_result"
    EXTERNAL = "external"


# ── Memory Types (§6) ───────────────────────────────────
class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    POLICY = "policy"
    META = "meta"              # reflection / system-generated memories (extension, not in spec §6)


# ── Proposal Status (§11.3) ─────────────────────────────
class ProposalStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    SIMULATED = "simulated"
    SCORED = "scored"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


# ── Intent types for retrieval (§10.2) ─────────────────
class IntentType(str, Enum):
    RECALL = "recall"
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"


# ── Belief Status (for TTM) ─────────────────────────────
class BeliefStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
