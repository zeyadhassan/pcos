"""Tests for identity resolution (identity_resolution.py)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from percos.engine.identity_resolution import IdentityResolver, entity_similarity
from percos.stores.tables import CommittedFactRow


class TestEntitySimilarity:
    def test_identical_names(self):
        assert entity_similarity("Alice", "Alice") == 1.0

    def test_completely_different(self):
        score = entity_similarity("Alice", "XYZ123")
        assert score < 0.5

    def test_similar_names(self):
        score = entity_similarity("Robert Smith", "Rob Smith")
        assert score > 0.5

    def test_case_insensitive(self):
        score = entity_similarity("alice", "Alice")
        assert score > 0.8

    def test_empty_strings(self):
        score = entity_similarity("", "")
        assert isinstance(score, float)


class TestIdentityResolver:
    @pytest.mark.asyncio
    async def test_find_duplicates_empty_db(self, db_session: AsyncSession):
        resolver = IdentityResolver(db_session)
        result = await resolver.find_duplicates()
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_find_duplicates_with_similar_entities(self, db_session: AsyncSession):
        now = datetime.now(tz=timezone.utc)
        # Create two similar entities
        for i, name in enumerate(["Robert Smith", "Rob Smith"]):
            row = CommittedFactRow(
                id=f"dup-{i}", candidate_id=f"c-{i}", entity_type="Person",
                entity_data={"name": name}, fact_type="observed",
                confidence="high", scope="global", sensitivity="public",
                source="test", created_at=now, last_verified=now,
                valid_from=now, belief_status="active", provenance_chain=[],
            )
            db_session.add(row)
        await db_session.flush()

        resolver = IdentityResolver(db_session, threshold=0.5)
        result = await resolver.find_duplicates(entity_type="Person")
        assert isinstance(result, list)
        # Should find at least one potential duplicate pair
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_find_duplicates_no_false_positives(self, db_session: AsyncSession):
        now = datetime.now(tz=timezone.utc)
        for i, name in enumerate(["Alice Johnson", "XYZ Corporation"]):
            row = CommittedFactRow(
                id=f"ndup-{i}", candidate_id=f"c-{i}", entity_type="Person",
                entity_data={"name": name}, fact_type="observed",
                confidence="high", scope="global", sensitivity="public",
                source="test", created_at=now, last_verified=now,
                valid_from=now, belief_status="active", provenance_chain=[],
            )
            db_session.add(row)
        await db_session.flush()

        resolver = IdentityResolver(db_session, threshold=0.85)
        result = await resolver.find_duplicates(entity_type="Person")
        # Very different names should not be flagged as duplicates
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_merge_entities(self, db_session: AsyncSession):
        now = datetime.now(tz=timezone.utc)
        for i, name in enumerate(["Robert Smith", "Rob Smith"]):
            row = CommittedFactRow(
                id=f"merge-{i}", candidate_id=f"c-{i}", entity_type="Person",
                entity_data={"name": name}, fact_type="observed",
                confidence="high", scope="global", sensitivity="public",
                source="test", created_at=now, last_verified=now,
                valid_from=now, belief_status="active", provenance_chain=[],
            )
            db_session.add(row)
        await db_session.flush()

        resolver = IdentityResolver(db_session)
        result = await resolver.merge("merge-0", ["merge-1"])
        assert isinstance(result, dict)
        assert "canonical_id" in result or "merged" in result or "canonical" in result

    @pytest.mark.asyncio
    async def test_suggest_canonical(self, db_session: AsyncSession):
        now = datetime.now(tz=timezone.utc)
        for i, (name, conf) in enumerate([("Robert Smith", "high"), ("Rob Smith", "low")]):
            row = CommittedFactRow(
                id=f"sug-{i}", candidate_id=f"c-{i}", entity_type="Person",
                entity_data={"name": name}, fact_type="observed",
                confidence=conf, scope="global", sensitivity="public",
                source="test", created_at=now, last_verified=now,
                valid_from=now, belief_status="active", provenance_chain=[],
            )
            db_session.add(row)
        await db_session.flush()

        resolver = IdentityResolver(db_session)
        canonical = await resolver.suggest_canonical(["sug-0", "sug-1"])
        assert canonical is not None
        assert canonical in ["sug-0", "sug-1"]
