"""Deadline & Recurrence Engine (§6.1).

Provides:
- ``check_upcoming_deadlines()`` — schema-driven: discovers entity types with
  deadline/start fields from the domain schema and scans active facts
- ``expand_recurrences()``       — RRULE string → concrete date occurrences
- ``get_reminders()``            — generate suggestions for upcoming/overdue items
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from percos.logging import get_logger

log = get_logger("deadlines")

# ── RRULE expansion ────────────────────────────────────

# Supported simple RRULE frequencies (subset – no BYDAY / INTERVAL yet)
_FREQ_DELTA = {
    "DAILY": timedelta(days=1),
    "WEEKLY": timedelta(weeks=1),
    "MONTHLY": timedelta(days=30),   # approximate
    "YEARLY": timedelta(days=365),
}


def expand_rrule(
    rrule: str,
    dtstart: datetime,
    after: datetime | None = None,
    before: datetime | None = None,
    max_occurrences: int = 50,
) -> list[datetime]:
    """Expand a simple RRULE string into concrete occurrences.

    Supports ``FREQ=DAILY|WEEKLY|MONTHLY|YEARLY`` and ``COUNT=N``.
    Falls back to returning ``[dtstart]`` for unsupported rules.
    """
    if not rrule:
        return [dtstart]

    parts: dict[str, str] = {}
    for segment in rrule.replace("RRULE:", "").split(";"):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k.upper()] = v.upper()

    freq = parts.get("FREQ", "")
    delta = _FREQ_DELTA.get(freq)
    if delta is None:
        log.debug("unsupported_rrule_freq", freq=freq, rrule=rrule)
        return [dtstart]

    count = int(parts.get("COUNT", str(max_occurrences)))
    count = min(count, max_occurrences)

    # Optional INTERVAL
    interval = int(parts.get("INTERVAL", "1"))
    if interval > 1:
        delta = delta * interval

    occurrences: list[datetime] = []
    current = dtstart
    for _ in range(count):
        if before and current > before:
            break
        if after is None or current >= after:
            occurrences.append(current)
        current = current + delta

    return occurrences


class DeadlineChecker:
    """Checks active task/event facts for upcoming deadlines and generates reminders."""

    def __init__(self, semantic_store):
        self._semantic = semantic_store

    async def check_upcoming_deadlines(
        self,
        horizon_days: int = 7,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return facts with deadlines within *horizon_days* from now.

        Entity types are discovered from the domain schema: any type with
        a ``deadline`` or ``start`` field is treated as deadline-bearing.

        Each result dict includes:
        ``fact_id``, ``entity_type``, ``name``, ``deadline``, ``days_left``,
        ``overdue`` (bool).
        """
        from percos.schema import get_domain_schema

        now = now or datetime.now(tz=timezone.utc)
        horizon = now + timedelta(days=horizon_days)

        results: list[dict[str, Any]] = []
        schema = get_domain_schema()

        # Discover which entity types have deadline-like or start-like fields
        deadline_types: list[tuple[str, list[str]]] = []
        for et in schema.entity_types.values():
            field_names = {f.name for f in et.fields}
            date_fields: list[str] = []
            for candidate in ("deadline", "start", "due_date", "end_date", "due_at"):
                if candidate in field_names:
                    date_fields.append(candidate)
            if date_fields:
                deadline_types.append((et.name.lower(), date_fields))

        for etype, date_fields in deadline_types:
            facts = await self._semantic.get_active_facts(entity_type=etype)
            for f in facts:
                data = f.entity_data or {}
                for date_field in date_fields:
                    raw_value = data.get(date_field)
                    dl = self._try_parse_dt(raw_value) if raw_value else None
                    if dl is None:
                        continue

                    # Expand recurrences if present
                    rrule = data.get("recurrence_rule")
                    if rrule:
                        occurrences = expand_rrule(rrule, dl, after=now, before=horizon)
                        for occ in occurrences:
                            days_left = (occ - now).total_seconds() / 86400
                            results.append({
                                "fact_id": f.id,
                                "entity_type": etype,
                                "name": data.get("name", ""),
                                "deadline": occ.isoformat(),
                                "days_left": round(days_left, 1),
                                "overdue": days_left < 0,
                                "recurrence": True,
                            })
                    elif dl <= horizon:
                        days_left = (dl - now).total_seconds() / 86400
                        results.append({
                            "fact_id": f.id,
                            "entity_type": etype,
                            "name": data.get("name", ""),
                            "deadline": dl.isoformat(),
                            "days_left": round(days_left, 1),
                            "overdue": days_left < 0,
                            "status": data.get("status", ""),
                        })
                    break  # only use the first matching date field per fact

        # Sort: overdue first, then by days_left ascending
        results.sort(key=lambda r: (not r.get("overdue", False), r.get("days_left", 999)))
        return results

    async def get_reminders(self, horizon_days: int = 7) -> list[dict[str, Any]]:
        """Generate actionable reminders from upcoming deadlines."""
        deadlines = await self.check_upcoming_deadlines(horizon_days)
        reminders: list[dict[str, Any]] = []
        for d in deadlines:
            if d.get("overdue"):
                urgency = "overdue"
                message = f"OVERDUE: {d['name']} was due {abs(d['days_left']):.0f} day(s) ago"
            elif d["days_left"] < 1:
                urgency = "urgent"
                message = f"Due TODAY: {d['name']}"
            elif d["days_left"] < 3:
                urgency = "soon"
                message = f"Due in {d['days_left']:.0f} day(s): {d['name']}"
            else:
                urgency = "upcoming"
                message = f"Coming up in {d['days_left']:.0f} day(s): {d['name']}"
            reminders.append({
                **d,
                "urgency": urgency,
                "message": message,
            })
        return reminders

    # ── Helpers ─────────────────────────────────────────

    @staticmethod
    def _parse_deadline(entity_data: dict) -> datetime | None:
        """Extract a deadline datetime from entity_data."""
        dl_str = entity_data.get("deadline")
        if not dl_str:
            return None
        return DeadlineChecker._try_parse_dt(dl_str)

    @staticmethod
    def _try_parse_dt(value) -> datetime | None:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if not isinstance(value, str):
            return None
        try:
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None
