"""Cross-device synchronisation protocol (§11.4 – GAP-L4).

Provides:
- Full / incremental **export** of the cognitive state (facts, episodes,
  procedures, policies) with a *version vector* for change tracking.
- **Import** with conflict-resolution strategies: ``newest_wins``,
  ``manual``, and ``merge``.
- Deterministic version-vector comparison for identifying which device
  has newer data.
"""

from __future__ import annotations

import hashlib
import json as _json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from percos.logging import get_logger
from percos.models.enums import Confidence, FactType, Sensitivity
from percos.stores.tables import (
    CommittedFactRow,
    EpisodicRow,
    ProceduralRow,
    PolicyRow,
)

log = get_logger("sync")

# Unique device id (generated once per runtime)
import platform as _platform

_DEVICE_ID = hashlib.md5(
    f"{_platform.node()}-{_platform.machine()}".encode()
).hexdigest()[:12]


def get_device_id() -> str:
    return _DEVICE_ID


class SyncProtocol:
    """Export / import cognitive state with change-tracking version vectors."""

    def __init__(self, session: AsyncSession):
        self._session = session

    # ── Export ──────────────────────────────────────────

    async def export_state(
        self,
        since: datetime | None = None,
        include_episodic: bool = True,
    ) -> dict[str, Any]:
        """Export facts, procedures, and policies (optionally eps) since a timestamp.

        Returns a serialisable dict with a version vector.
        """
        # Compute schema fingerprint for compatibility checks
        try:
            from percos.schema import get_domain_schema
            schema = get_domain_schema()
            schema_version = schema.version
            schema_hash = hashlib.md5(
                _json.dumps(sorted(schema.get_model_map().keys())).encode()
            ).hexdigest()[:12]
        except Exception:
            schema_version = "unknown"
            schema_hash = "unknown"

        result: dict[str, Any] = {
            "device_id": _DEVICE_ID,
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            "schema_version": schema_version,
            "schema_hash": schema_hash,
            "version_vector": {},
            "facts": [],
            "episodic": [],
            "procedures": [],
            "policies": [],
        }

        # ── Facts ──
        stmt = select(CommittedFactRow).order_by(CommittedFactRow.created_at)
        if since:
            stmt = stmt.where(CommittedFactRow.created_at >= since)
        rows = (await self._session.execute(stmt)).scalars().all()
        for r in rows:
            result["facts"].append({
                "id": r.id,
                "entity_type": r.entity_type,
                "entity_data": r.entity_data,
                "fact_type": r.fact_type,
                "confidence": r.confidence,
                "scope": r.scope,
                "sensitivity": r.sensitivity,
                "source": r.source,
                "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                "valid_to": r.valid_to.isoformat() if r.valid_to else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "last_verified": r.last_verified.isoformat() if getattr(r, "last_verified", None) else None,
            })

        # ── Episodic ──
        if include_episodic:
            stmt_e = select(EpisodicRow).order_by(EpisodicRow.timestamp)
            if since:
                stmt_e = stmt_e.where(EpisodicRow.timestamp >= since)
            epi_rows = (await self._session.execute(stmt_e)).scalars().all()
            for r in epi_rows:
                result["episodic"].append({
                    "id": r.id,
                    "event_id": r.event_id,
                    "memory_type": r.memory_type,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "content": r.content,
                    "metadata_extra": r.metadata_extra,
                })

        # ── Procedures ──
        stmt_p = select(ProceduralRow)
        proc_rows = (await self._session.execute(stmt_p)).scalars().all()
        for r in proc_rows:
            result["procedures"].append({
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "steps": r.steps,
                "trigger": r.trigger,
                "version": r.version,
                "success_rate": r.success_rate,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })

        # ── Policies ──
        stmt_pol = select(PolicyRow)
        pol_rows = (await self._session.execute(stmt_pol)).scalars().all()
        for r in pol_rows:
            result["policies"].append({
                "id": r.id,
                "name": r.name,
                "rule": r.rule,
                "effect": r.effect,
                "priority": r.priority,
                "scope": r.scope,
                "active": r.active,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })

        # Version vector: counts per type
        result["version_vector"] = {
            "device": _DEVICE_ID,
            "facts": len(result["facts"]),
            "episodic": len(result["episodic"]),
            "procedures": len(result["procedures"]),
            "policies": len(result["policies"]),
            "timestamp": result["exported_at"],
        }

        log.info(
            "state_exported",
            facts=len(result["facts"]),
            episodic=len(result["episodic"]),
            procedures=len(result["procedures"]),
            policies=len(result["policies"]),
        )
        return result

    # ── Import ──────────────────────────────────────────

    async def import_state(
        self,
        payload: dict[str, Any],
        strategy: str = "newest_wins",
    ) -> dict[str, Any]:
        """Import state from another device.

        Strategies:
        - ``newest_wins``: If a fact with the same ID exists, keep the newer one.
        - ``manual``: Skip conflicts and return them for user review.
        - ``merge``: Import everything, creating new IDs for conflicts.

        Returns summary with imported / skipped / conflict counts.
        """
        imported = 0
        skipped = 0
        conflicts: list[dict[str, Any]] = []
        warnings: list[str] = []

        # ── Schema version compatibility check ──
        remote_schema_hash = payload.get("schema_hash", "")
        remote_schema_version = payload.get("schema_version", "")
        try:
            from percos.schema import get_domain_schema
            schema = get_domain_schema()
            local_hash = hashlib.md5(
                _json.dumps(sorted(schema.get_model_map().keys())).encode()
            ).hexdigest()[:12]
            if remote_schema_hash and remote_schema_hash != local_hash:
                warnings.append(
                    f"Schema mismatch: remote hash {remote_schema_hash} "
                    f"(v{remote_schema_version}) != local hash {local_hash} "
                    f"(v{schema.version}). Some entity data may not conform."
                )
                log.warning(
                    "sync_schema_mismatch",
                    remote_hash=remote_schema_hash,
                    local_hash=local_hash,
                )
        except Exception:
            pass

        # ── Facts ──
        for fact_data in payload.get("facts", []):
            fid = fact_data.get("id", "")
            existing = (await self._session.execute(
                select(CommittedFactRow).where(CommittedFactRow.id == fid)
            )).scalar_one_or_none()

            if existing is None:
                row = CommittedFactRow(
                    id=fid or str(uuid4()),
                    entity_type=fact_data.get("entity_type", ""),
                    entity_data=fact_data.get("entity_data", {}),
                    fact_type=fact_data.get("fact_type", "observed"),
                    confidence=fact_data.get("confidence", "medium"),
                    scope=fact_data.get("scope", "global"),
                    sensitivity=fact_data.get("sensitivity", "internal"),
                    source=fact_data.get("source", "sync_import"),
                )
                if fact_data.get("valid_from"):
                    row.valid_from = datetime.fromisoformat(fact_data["valid_from"])
                if fact_data.get("valid_to"):
                    row.valid_to = datetime.fromisoformat(fact_data["valid_to"])
                self._session.add(row)
                imported += 1
            else:
                # Conflict
                if strategy == "newest_wins":
                    remote_ts = fact_data.get("created_at", "")
                    local_ts = existing.created_at.isoformat() if existing.created_at else ""
                    if remote_ts > local_ts:
                        # Remote is newer – update local
                        existing.entity_data = fact_data.get("entity_data", existing.entity_data)
                        existing.confidence = fact_data.get("confidence", existing.confidence)
                        existing.source = fact_data.get("source", existing.source)
                        imported += 1
                    else:
                        skipped += 1
                elif strategy == "manual":
                    conflicts.append({
                        "type": "fact",
                        "id": fid,
                        "local": {
                            "entity_data": existing.entity_data,
                            "created_at": existing.created_at.isoformat() if existing.created_at else None,
                        },
                        "remote": fact_data,
                    })
                    skipped += 1
                elif strategy == "merge":
                    new_id = str(uuid4())
                    row = CommittedFactRow(
                        id=new_id,
                        entity_type=fact_data.get("entity_type", ""),
                        entity_data=fact_data.get("entity_data", {}),
                        fact_type=fact_data.get("fact_type", "observed"),
                        confidence=fact_data.get("confidence", "medium"),
                        scope=fact_data.get("scope", "global"),
                        sensitivity=fact_data.get("sensitivity", "internal"),
                        source=f"sync_merge:{fact_data.get('source', '')}",
                    )
                    self._session.add(row)
                    imported += 1
                else:
                    skipped += 1

        # ── Episodic ──
        for epi in payload.get("episodic", []):
            eid = epi.get("id", "")
            existing = (await self._session.execute(
                select(EpisodicRow).where(EpisodicRow.id == eid)
            )).scalar_one_or_none()

            if existing is None:
                row = EpisodicRow(
                    id=eid or str(uuid4()),
                    event_id=epi.get("event_id", ""),
                    memory_type=epi.get("memory_type", "episodic"),
                    content=epi.get("content", ""),
                    metadata_extra=epi.get("metadata_extra", {}),
                )
                if epi.get("timestamp"):
                    row.timestamp = datetime.fromisoformat(epi["timestamp"])
                self._session.add(row)
                imported += 1
            else:
                # Episodic memories are append-only; skip duplicates
                skipped += 1

        # ── Procedures ──
        for proc in payload.get("procedures", []):
            pid = proc.get("id", "")
            existing = (await self._session.execute(
                select(ProceduralRow).where(ProceduralRow.id == pid)
            )).scalar_one_or_none()

            if existing is None:
                row = ProceduralRow(
                    id=pid or str(uuid4()),
                    name=proc.get("name", ""),
                    description=proc.get("description", ""),
                    steps=proc.get("steps", []),
                    trigger=proc.get("trigger", ""),
                    version=proc.get("version", 1),
                    success_rate=proc.get("success_rate"),
                )
                self._session.add(row)
                imported += 1
            else:
                remote_v = proc.get("version", 0)
                if remote_v > (existing.version or 0):
                    existing.steps = proc.get("steps", existing.steps)
                    existing.description = proc.get("description", existing.description)
                    existing.version = remote_v
                    imported += 1
                else:
                    skipped += 1

        # ── Policies ──
        for pol in payload.get("policies", []):
            pol_id = pol.get("id", "")
            existing = (await self._session.execute(
                select(PolicyRow).where(PolicyRow.id == pol_id)
            )).scalar_one_or_none()

            if existing is None:
                row = PolicyRow(
                    id=pol_id or str(uuid4()),
                    name=pol.get("name", ""),
                    rule=pol.get("rule", ""),
                    effect=pol.get("effect", "deny"),
                    priority=pol.get("priority", 0),
                    scope=pol.get("scope", "global"),
                    active=pol.get("active", True),
                )
                self._session.add(row)
                imported += 1
            else:
                skipped += 1

        await self._session.flush()
        log.info("state_imported", imported=imported, skipped=skipped, conflicts=len(conflicts))

        return {
            "imported": imported,
            "skipped": skipped,
            "conflicts": conflicts,
            "warnings": warnings,
            "device_id": _DEVICE_ID,
        }


# ── Peer Sync Client (v3 scaffolding) ─────────────────

class PeerSyncClient:
    """HTTP-based peer sync transport.

    Pushes local state to and pulls remote state from another PERCOS
    instance over its REST API.  This is v3 scaffolding — production use
    would add mTLS, high-watermark tracking, and automatic scheduling.
    """

    def __init__(
        self,
        session: AsyncSession,
        peer_url: str,
        *,
        strategy: str = "newest_wins",
        timeout: int = 30,
    ):
        self._sync = SyncProtocol(session)
        self._peer_url = peer_url.rstrip("/")
        self._strategy = strategy
        self._timeout = timeout

    async def push(self, since: datetime | None = None) -> dict[str, Any]:
        """Export local state and POST it to the peer's /sync/import endpoint."""
        import httpx

        payload = await self._sync.export_state(since=since, include_episodic=True)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._peer_url}/sync/import",
                json=payload,
                params={"strategy": self._strategy},
            )
            resp.raise_for_status()
            result = resp.json()
        log.info("peer_push_complete", peer=self._peer_url, result=result)
        return result

    async def pull(self, since: datetime | None = None) -> dict[str, Any]:
        """GET remote state from the peer and import it locally."""
        import httpx

        params: dict[str, str] = {}
        if since:
            params["since"] = since.isoformat()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._peer_url}/sync/export",
                params=params,
            )
            resp.raise_for_status()
            remote_payload = resp.json()

        result = await self._sync.import_state(
            remote_payload, strategy=self._strategy,
        )
        log.info("peer_pull_complete", peer=self._peer_url, result=result)
        return result

    async def bidirectional_sync(
        self, since: datetime | None = None,
    ) -> dict[str, Any]:
        """Pull then push — ensures both sides converge."""
        pull_result = await self.pull(since=since)
        push_result = await self.push(since=since)
        return {
            "pull": pull_result,
            "push": push_result,
        }
