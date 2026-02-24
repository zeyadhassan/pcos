"""Centralized audit log (§11.2, §17 principle 8).

Provides a unified trail for all system actions: fact commits, belief edits,
evolution deployments, policy changes, maintenance runs, etc.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from percos.stores.tables import AuditLogRow


class AuditLog:
    """Append-only audit trail for all PCOS actions."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def record(
        self,
        action: str,
        component: str,
        *,
        actor: str = "system",
        resource_id: str = "",
        resource_type: str = "",
        details: dict[str, Any] | None = None,
        outcome: str = "success",
    ) -> str:
        """Record an auditable action. Returns the audit entry ID."""
        row = AuditLogRow(
            action=action,
            component=component,
            actor=actor,
            resource_id=resource_id,
            resource_type=resource_type,
            details=details or {},
            outcome=outcome,
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def query(
        self,
        *,
        action: str | None = None,
        component: str | None = None,
        resource_id: str | None = None,
        actor: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query the audit log with optional filters."""
        stmt = select(AuditLogRow)
        if action:
            stmt = stmt.where(AuditLogRow.action == action)
        if component:
            stmt = stmt.where(AuditLogRow.component == component)
        if resource_id:
            stmt = stmt.where(AuditLogRow.resource_id == resource_id)
        if actor:
            stmt = stmt.where(AuditLogRow.actor == actor)
        if since:
            stmt = stmt.where(AuditLogRow.timestamp >= since)
        stmt = stmt.order_by(AuditLogRow.timestamp.desc()).limit(limit)
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "action": r.action,
                "component": r.component,
                "actor": r.actor,
                "resource_id": r.resource_id,
                "resource_type": r.resource_type,
                "details": r.details,
                "outcome": r.outcome,
            }
            for r in rows
        ]

    async def count(self, action: str | None = None) -> int:
        """Count audit entries, optionally filtered by action."""
        stmt = select(func.count()).select_from(AuditLogRow)
        if action:
            stmt = stmt.where(AuditLogRow.action == action)
        result = await self._session.execute(stmt)
        return result.scalar_one()
