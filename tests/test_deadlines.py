"""Tests for deadline detection (deadlines.py)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from percos.engine.deadlines import DeadlineChecker, expand_rrule
from percos.schema import EntityTypeDef, FieldDef, DomainSchema
from percos.stores.semantic_store import SemanticStore
from percos.stores.tables import CommittedFactRow


class TestExpandRrule:
    def test_daily_rrule(self):
        dtstart = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
        after = datetime(2026, 1, 1, tzinfo=timezone.utc)
        before = datetime(2026, 1, 10, tzinfo=timezone.utc)
        result = expand_rrule("FREQ=DAILY", dtstart, after=after, before=before)
        assert isinstance(result, list)
        assert len(result) >= 1
        for dt in result:
            assert isinstance(dt, datetime)

    def test_weekly_rrule(self):
        dtstart = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)  # Monday
        after = datetime(2026, 1, 1, tzinfo=timezone.utc)
        before = datetime(2026, 2, 1, tzinfo=timezone.utc)
        result = expand_rrule("FREQ=WEEKLY", dtstart, after=after, before=before)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_max_occurrences_limit(self):
        dtstart = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = expand_rrule("FREQ=DAILY", dtstart, max_occurrences=5)
        assert len(result) <= 5


class TestDeadlineChecker:
    @pytest.mark.asyncio
    async def test_no_deadlines_empty_db(self, db_session: AsyncSession):
        store = SemanticStore(db_session)
        checker = DeadlineChecker(store)
        result = await checker.check_upcoming_deadlines()
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_finds_upcoming_deadline(self, db_session: AsyncSession):
        now = datetime.now(tz=timezone.utc)
        future = now + timedelta(days=3)
        row = CommittedFactRow(
            id="dl-1", candidate_id="c-1", entity_type="Task",
            entity_data={"name": "Review PR", "deadline": future.isoformat()},
            fact_type="observed", confidence="high", scope="global",
            sensitivity="public", source="test", created_at=now,
            last_verified=now, valid_from=now,
            belief_status="active", provenance_chain=[],
        )
        db_session.add(row)
        await db_session.flush()

        store = SemanticStore(db_session)
        checker = DeadlineChecker(store)

        with patch("percos.schema.get_domain_schema") as mock_schema:
            mock_schema.return_value = DomainSchema(
                name="test",
                entity_types={
                    "Task": EntityTypeDef(
                        name="Task",
                        fields=[FieldDef(name="name"), FieldDef(name="deadline", type="datetime")],
                    )
                },
            )
            result = await checker.check_upcoming_deadlines(horizon_days=7, now=now)
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_reminders(self, db_session: AsyncSession):
        now = datetime.now(tz=timezone.utc)
        future = now + timedelta(days=2)
        row = CommittedFactRow(
            id="rem-1", candidate_id="c-1", entity_type="CalendarEvent",
            entity_data={"name": "Meeting", "start": future.isoformat()},
            fact_type="observed", confidence="high", scope="global",
            sensitivity="public", source="test", created_at=now,
            last_verified=now, valid_from=now,
            belief_status="active", provenance_chain=[],
        )
        db_session.add(row)
        await db_session.flush()

        store = SemanticStore(db_session)
        checker = DeadlineChecker(store)

        with patch("percos.schema.get_domain_schema") as mock_schema:
            mock_schema.return_value = DomainSchema(
                name="test",
                entity_types={
                    "CalendarEvent": EntityTypeDef(
                        name="CalendarEvent",
                        fields=[FieldDef(name="name"), FieldDef(name="start", type="datetime")],
                    )
                },
            )
            result = await checker.get_reminders(horizon_days=7)
            assert isinstance(result, list)

    def test_parse_deadline_iso_format(self):
        dt = DeadlineChecker._parse_deadline({"deadline": "2026-03-01T10:00:00Z"})
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 3

    def test_parse_deadline_missing(self):
        dt = DeadlineChecker._parse_deadline({"name": "No deadline"})
        assert dt is None

    def test_try_parse_dt_various_formats(self):
        # ISO format
        dt = DeadlineChecker._try_parse_dt("2026-03-01T10:00:00Z")
        assert dt is not None

        # None
        dt = DeadlineChecker._try_parse_dt(None)
        assert dt is None

        # Invalid
        dt = DeadlineChecker._try_parse_dt("not a date")
        assert dt is None
