"""Memory security layer (§11) – defenses against poisoning and misuse.

Implements:
- Confidence threshold enforcement for auto-accept
- Input sanitization against prompt injection
- Rate limiting per source
- Sensitivity-aware access control
- GAP-L3: Fine-grained redaction engine for partial data masking
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from percos.logging import get_logger
from percos.models.enums import Confidence, Sensitivity

log = get_logger("security")

# ── Confidence thresholds ────────────────────────────────
# Minimum confidence required for auto-commit (Gap #13)
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
MIN_AUTO_COMMIT_CONFIDENCE = "medium"  # must be >= this to auto-commit


def meets_confidence_threshold(
    confidence: str | Confidence,
    threshold: str = MIN_AUTO_COMMIT_CONFIDENCE,
) -> bool:
    """Check if a confidence level meets the minimum threshold."""
    conf_val = confidence.value if isinstance(confidence, Confidence) else confidence
    return CONFIDENCE_ORDER.get(conf_val, 0) >= CONFIDENCE_ORDER.get(threshold, 1)


# ── Input sanitization ──────────────────────────────────
# Patterns that suggest prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s*:\s*you\s+are",
    r"forget\s+(everything|all)",
    r"new\s+instructions?\s*:",
    r"override\s+(all\s+)?(rules|policies|instructions)",
    r"\bDAN\b.*\bjailbreak\b",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def sanitize_input(text: str) -> tuple[str, list[str]]:
    """Sanitize input text and return (cleaned_text, warnings).

    Returns the original text and any injection warnings found.
    We don't strip content (that could lose valid data) but flag suspicious patterns.
    """
    warnings: list[str] = []
    for pattern in _compiled_patterns:
        if pattern.search(text):
            warnings.append(f"Suspicious pattern detected: {pattern.pattern}")
    if warnings:
        log.warning("potential_injection", warning_count=len(warnings), text_preview=text[:100])
    return text, warnings


def is_safe_input(text: str) -> bool:
    """Quick check if input appears safe."""
    _, warnings = sanitize_input(text)
    return len(warnings) == 0


# ── Rate limiting ────────────────────────────────────────

class RateLimiter:
    """Simple in-memory rate limiter per source key."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        """Return True if request is allowed, False if rate limited."""
        now = time.monotonic()
        # Clean old entries
        self._requests[key] = [
            t for t in self._requests[key] if now - t < self._window
        ]
        if len(self._requests[key]) >= self._max:
            log.warning("rate_limited", key=key, count=len(self._requests[key]))
            return False
        self._requests[key].append(now)
        return True

    def reset(self, key: str | None = None) -> None:
        if key:
            self._requests.pop(key, None)
        else:
            self._requests.clear()


# Module-level rate limiter
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


# ── Sensitivity access control ───────────────────────────

SENSITIVITY_LEVELS = {"public": 0, "internal": 1, "private": 2, "secret": 3}


def check_sensitivity_access(
    fact_sensitivity: str | Sensitivity,
    requester_clearance: str = "internal",
) -> bool:
    """Check if requester has clearance to access a fact at given sensitivity."""
    sens = fact_sensitivity.value if isinstance(fact_sensitivity, Sensitivity) else fact_sensitivity
    return SENSITIVITY_LEVELS.get(sens, 0) <= SENSITIVITY_LEVELS.get(requester_clearance, 1)


# ── GAP-L3: Redaction Engine ────────────────────────────

# Built-in PII patterns for automatic redaction
BUILTIN_PII_PATTERNS: dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone_us": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "phone_intl": r"\b\+\d{1,3}[-.\s]?\d{4,14}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "date_of_birth": r"\b(?:DOB|Date of Birth|Born)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
    "passport": r"\b[A-Z]{1,2}\d{6,9}\b",
}


@dataclass
class RedactionRule:
    """A named redaction rule with a regex pattern and replacement text."""
    name: str
    pattern: str
    replacement: str = "[REDACTED]"
    sensitivity_level: str = "private"  # minimum sensitivity level to trigger
    _compiled: re.Pattern | None = field(default=None, repr=False, compare=False)

    @property
    def compiled(self) -> re.Pattern:
        if self._compiled is None:
            self._compiled = re.compile(self.pattern, re.IGNORECASE)
        return self._compiled


class RedactionEngine:
    """Fine-grained redaction engine for partial data masking (GAP-L3).

    Supports:
    - Built-in PII patterns (email, phone, SSN, credit card, etc.)
    - Custom user-defined rules
    - Sensitivity-aware redaction (only applies if content meets threshold)
    - Field-level redaction for structured data
    """

    def __init__(self, enable_builtin_pii: bool = True):
        self._rules: dict[str, RedactionRule] = {}
        if enable_builtin_pii:
            for name, pattern in BUILTIN_PII_PATTERNS.items():
                self._rules[name] = RedactionRule(
                    name=name,
                    pattern=pattern,
                    replacement=f"[{name.upper()}_REDACTED]",
                    sensitivity_level="internal",
                )

    def add_rule(
        self,
        name: str,
        pattern: str,
        replacement: str = "[REDACTED]",
        sensitivity_level: str = "private",
    ) -> None:
        """Add a custom redaction rule."""
        self._rules[name] = RedactionRule(
            name=name,
            pattern=pattern,
            replacement=replacement,
            sensitivity_level=sensitivity_level,
        )
        log.info("redaction_rule_added", name=name)

    def remove_rule(self, name: str) -> bool:
        """Remove a redaction rule by name."""
        return self._rules.pop(name, None) is not None

    def list_rules(self) -> list[dict[str, str]]:
        """List all active redaction rules."""
        return [
            {
                "name": r.name,
                "pattern": r.pattern,
                "replacement": r.replacement,
                "sensitivity_level": r.sensitivity_level,
            }
            for r in self._rules.values()
        ]

    def redact(
        self,
        text: str,
        requester_clearance: str = "internal",
    ) -> tuple[str, list[str]]:
        """Apply redaction rules to text based on requester clearance.

        Only applies rules whose ``sensitivity_level`` is above the
        requester's clearance (i.e., content that the requester should
        not see in full detail).

        Returns:
            Tuple of (redacted_text, list_of_rule_names_applied).
        """
        applied: list[str] = []
        result = text
        req_level = SENSITIVITY_LEVELS.get(requester_clearance, 1)

        for rule in self._rules.values():
            rule_level = SENSITIVITY_LEVELS.get(rule.sensitivity_level, 2)
            # Apply rule only if the content sensitivity exceeds clearance
            if rule_level > req_level:
                new_text = rule.compiled.sub(rule.replacement, result)
                if new_text != result:
                    applied.append(rule.name)
                    result = new_text

        if applied:
            log.info("text_redacted", rules_applied=len(applied), clearance=requester_clearance)
        return result, applied

    def redact_dict(
        self,
        data: dict[str, Any],
        requester_clearance: str = "internal",
        fields: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Apply redaction to string values in a dictionary.

        Args:
            data: Dictionary to redact.
            requester_clearance: Requester's sensitivity clearance.
            fields: If provided, only redact these specific keys.
                    If None, redact all string values.

        Returns:
            Tuple of (redacted_dict, list_of_rule_names_applied).
        """
        all_applied: list[str] = []
        result = dict(data)

        for key, value in result.items():
            if fields and key not in fields:
                continue
            if isinstance(value, str):
                redacted, applied = self.redact(value, requester_clearance)
                result[key] = redacted
                all_applied.extend(applied)
            elif isinstance(value, dict):
                redacted_dict, applied = self.redact_dict(value, requester_clearance, fields)
                result[key] = redacted_dict
                all_applied.extend(applied)

        return result, list(set(all_applied))

    def detect_pii(self, text: str) -> list[dict[str, str]]:
        """Detect PII in text without redacting. Returns list of findings."""
        findings: list[dict[str, str]] = []
        for rule in self._rules.values():
            matches = rule.compiled.findall(text)
            for match in matches:
                match_str = match if isinstance(match, str) else match[0]
                findings.append({
                    "rule": rule.name,
                    "match": match_str[:4] + "..." if len(match_str) > 4 else match_str,
                    "category": rule.name,
                })
        return findings


# Module-level singleton
_redaction_engine: RedactionEngine | None = None


def get_redaction_engine() -> RedactionEngine:
    """Return the singleton redaction engine."""
    global _redaction_engine
    if _redaction_engine is None:
        _redaction_engine = RedactionEngine()
    return _redaction_engine
