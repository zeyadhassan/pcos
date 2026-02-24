"""Tests for the cross-device sync protocol (sync.py)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from percos.engine.sync import SyncProtocol, get_device_id
from percos.stores.tables import CommittedFactRow, EpisodicRow


class TestDeviceId:
    def test_get_device_id_returns_string(self):
        device_id = get_device_id()
        assert isinstance(device_id, str)
        assert len(device_id) > 0

    def test_get_device_id_stable(self):
        """Device ID should be deterministic for the same machine."""
        assert get_device_id() == get_device_id()


class TestSyncProtocolExport:
    @pytest.mark.asyncio
    async def test_export_empty_state(self, db_session: AsyncSession):
        proto = SyncProtocol(db_session)
        result = await proto.export_state()
        assert isinstance(result, dict)
        assert "device_id" in result
        assert "facts" in result

    @pytest.mark.asyncio
    async def test_export_includes_committed_facts(self, db_session: AsyncSession):
        now = datetime.now(tz=timezone.utc)
        row = CommittedFactRow(
            id="fact-1",
            candidate_id="cand-1",
            entity_type="Person",
            entity_data={"name": "Alice"},
            fact_type="observed",
            confidence="high",
            scope="global",
            sensitivity="public",
            source="test",
            created_at=now,
            last_verified=now,
            valid_from=now,
            belief_status="active",
            provenance_chain=[],
        )
        db_session.add(row)
        await db_session.flush()

        proto = SyncProtocol(db_session)
        result = await proto.export_state()
        assert len(result["facts"]) >= 1

    @pytest.mark.asyncio
    async def test_export_since_filter(self, db_session: AsyncSession):
        old = datetime.now(tz=timezone.utc) - timedelta(days=10)
        recent = datetime.now(tz=timezone.utc)
        row_old = CommittedFactRow(
            id="old-1", candidate_id="c-1", entity_type="Person",
            entity_data={"name": "Old"}, fact_type="observed",
            confidence="high", scope="global", sensitivity="public",
            source="test", created_at=old, last_verified=old,
            valid_from=old, belief_status="active", provenance_chain=[],
        )
        row_new = CommittedFactRow(
            id="new-1", candidate_id="c-2", entity_type="Person",
            entity_data={"name": "New"}, fact_type="observed",
            confidence="high", scope="global", sensitivity="public",
            source="test", created_at=recent, last_verified=recent,
            valid_from=recent, belief_status="active", provenance_chain=[],
        )
        db_session.add_all([row_old, row_new])
        await db_session.flush()

        proto = SyncProtocol(db_session)
        since = datetime.now(tz=timezone.utc) - timedelta(days=5)
        result = await proto.export_state(since=since)
        fact_ids = [f["id"] for f in result["facts"]]
        assert "new-1" in fact_ids


class TestSyncProtocolImport:
    @pytest.mark.asyncio
    async def test_import_empty_payload(self, db_session: AsyncSession):
        proto = SyncProtocol(db_session)
        payload = {
            "device_id": "remote-device",
            "facts": [],
            "episodic": [],
            "procedures": [],
            "policies": [],
            "version_vector": {},
        }
        result = await proto.import_state(payload)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_roundtrip_export_import(self, db_session: AsyncSession):
        """Export from one session and import into a fresh state."""
        now = datetime.now(tz=timezone.utc)
        row = CommittedFactRow(
            id="rt-1", candidate_id="c-1", entity_type="Task",
            entity_data={"name": "Deploy"}, fact_type="observed",
            confidence="high", scope="global", sensitivity="public",
            source="test", created_at=now, last_verified=now,
            valid_from=now, belief_status="active", provenance_chain=[],
        )
        db_session.add(row)
        await db_session.flush()

        proto = SyncProtocol(db_session)
        exported = await proto.export_state()
        assert len(exported["facts"]) >= 1

        # Import same payload (should not crash; idempotent or skip duplicates)
        result = await proto.import_state(exported, strategy="newest_wins")
        assert isinstance(result, dict)
