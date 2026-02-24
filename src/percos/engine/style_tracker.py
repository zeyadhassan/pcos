"""Communication Style Tracker (§14.3).

Learns user communication style over time by analysing messages for traits
such as formality, verbosity, emoji usage, sentence complexity, and tone.
The learned profile is stored in working memory and optionally persisted as
a committed fact so it survives across sessions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from percos.logging import get_logger

log = get_logger("style_tracker")

# ── Heuristic thresholds ───────────────────────────────
_EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F900-\U0001F9FF"
    r"\U00002702-\U000027B0"
    r"\U0000FE00-\U0000FE0F"
    r"\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF"
    r"\U00002600-\U000026FF]+",
    flags=re.UNICODE,
)
_FORMAL_MARKERS = {"please", "kindly", "sincerely", "regards", "furthermore", "therefore", "additionally"}
_INFORMAL_MARKERS = {"hey", "hi", "lol", "haha", "gonna", "wanna", "yep", "nope", "btw", "omg"}


@dataclass
class StyleProfile:
    """Aggregated communication style profile."""

    messages_analysed: int = 0
    avg_word_count: float = 0.0
    avg_sentence_length: float = 0.0
    emoji_ratio: float = 0.0            # fraction of messages containing emoji
    formality_score: float = 0.5        # 0 = very informal, 1 = very formal
    question_ratio: float = 0.0         # fraction of messages that are questions
    exclamation_ratio: float = 0.0      # fraction containing '!'
    greeting_ratio: float = 0.0         # fraction starting with greeting

    # Running counters (not exposed in dict serialisation)
    _total_words: int = field(default=0, repr=False)
    _total_sentences: int = field(default=0, repr=False)
    _emoji_messages: int = field(default=0, repr=False)
    _question_messages: int = field(default=0, repr=False)
    _exclamation_messages: int = field(default=0, repr=False)
    _greeting_messages: int = field(default=0, repr=False)
    _formal_hits: int = field(default=0, repr=False)
    _informal_hits: int = field(default=0, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages_analysed": self.messages_analysed,
            "avg_word_count": round(self.avg_word_count, 2),
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "emoji_ratio": round(self.emoji_ratio, 3),
            "formality_score": round(self.formality_score, 3),
            "question_ratio": round(self.question_ratio, 3),
            "exclamation_ratio": round(self.exclamation_ratio, 3),
            "greeting_ratio": round(self.greeting_ratio, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StyleProfile":
        """Reconstruct profile from serialised dict (partial restore)."""
        profile = cls()
        profile.messages_analysed = data.get("messages_analysed", 0)
        profile.avg_word_count = data.get("avg_word_count", 0.0)
        profile.avg_sentence_length = data.get("avg_sentence_length", 0.0)
        profile.emoji_ratio = data.get("emoji_ratio", 0.0)
        profile.formality_score = data.get("formality_score", 0.5)
        profile.question_ratio = data.get("question_ratio", 0.0)
        profile.exclamation_ratio = data.get("exclamation_ratio", 0.0)
        profile.greeting_ratio = data.get("greeting_ratio", 0.0)
        # Reconstruct running counters approximately
        n = max(profile.messages_analysed, 1)
        profile._total_words = int(profile.avg_word_count * n)
        profile._total_sentences = int(profile.avg_sentence_length * n) if profile.avg_sentence_length else n
        profile._emoji_messages = int(profile.emoji_ratio * n)
        profile._question_messages = int(profile.question_ratio * n)
        profile._exclamation_messages = int(profile.exclamation_ratio * n)
        profile._greeting_messages = int(profile.greeting_ratio * n)
        return profile


class CommunicationStyleTracker:
    """Analyses user messages and maintains a running style profile."""

    def __init__(self) -> None:
        self.profile = StyleProfile()

    def analyse(self, message: str) -> dict[str, Any]:
        """Analyse a single message and update the running profile.

        Returns a per-message feature dict.
        """
        words = message.split()
        word_count = len(words)
        sentences = [s.strip() for s in re.split(r"[.!?]+", message) if s.strip()]
        sentence_count = max(len(sentences), 1)
        has_emoji = bool(_EMOJI_RE.search(message))
        is_question = message.rstrip().endswith("?")
        has_exclamation = "!" in message

        lower_words = {w.lower().strip(",.!?;:") for w in words}
        formal_count = len(lower_words & _FORMAL_MARKERS)
        informal_count = len(lower_words & _INFORMAL_MARKERS)

        first_word = words[0].lower().strip(",.!?;:") if words else ""
        is_greeting = first_word in {"hi", "hey", "hello", "morning", "evening", "afternoon"}

        # Update running counters
        p = self.profile
        p.messages_analysed += 1
        p._total_words += word_count
        p._total_sentences += sentence_count
        if has_emoji:
            p._emoji_messages += 1
        if is_question:
            p._question_messages += 1
        if has_exclamation:
            p._exclamation_messages += 1
        if is_greeting:
            p._greeting_messages += 1
        p._formal_hits += formal_count
        p._informal_hits += informal_count

        n = p.messages_analysed
        p.avg_word_count = p._total_words / n
        p.avg_sentence_length = p._total_words / max(p._total_sentences, 1)
        p.emoji_ratio = p._emoji_messages / n
        p.question_ratio = p._question_messages / n
        p.exclamation_ratio = p._exclamation_messages / n
        p.greeting_ratio = p._greeting_messages / n

        # Formality: weighted running score between 0 and 1
        total_markers = p._formal_hits + p._informal_hits
        if total_markers > 0:
            p.formality_score = p._formal_hits / total_markers
        else:
            # Heuristic: longer sentences and higher word count → more formal
            p.formality_score = min(1.0, max(0.0, 0.3 + (p.avg_sentence_length - 5) * 0.05))

        features = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "has_emoji": has_emoji,
            "is_question": is_question,
            "has_exclamation": has_exclamation,
            "is_greeting": is_greeting,
            "formal_markers": formal_count,
            "informal_markers": informal_count,
        }
        log.debug("style_analysed", features=features, profile_n=n)
        return features

    def get_profile(self) -> dict[str, Any]:
        """Return the current style profile as a serialisable dict."""
        return self.profile.to_dict()

    def load_profile(self, data: dict[str, Any]) -> None:
        """Restore profile from a previously serialised dict."""
        self.profile = StyleProfile.from_dict(data)
