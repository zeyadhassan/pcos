"""Procedural and policy stores."""

from __future__ import annotations

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from percos.models.events import PolicyEntry, ProceduralEntry
from percos.stores.tables import PolicyRow, ProceduralRow, ProceduralVersionRow


class ProceduralStore:
    """Manages skills / workflows / templates with full versioning (§4.A.3)."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, entry: ProceduralEntry) -> str:
        row = ProceduralRow(
            id=str(entry.id),
            name=entry.name,
            description=entry.description,
            steps=entry.steps,
            trigger=entry.trigger,
            version=entry.version,
            success_rate=entry.success_rate,
            metadata_extra=entry.metadata_extra,
        )
        self._session.add(row)
        await self._session.flush()
        # Save initial version snapshot
        await self._snapshot_version(row)
        return str(entry.id)

    async def update(self, entry_id: str, updates: dict) -> ProceduralRow | None:
        """Update a procedure with automatic version increment and history.

        Returns the updated row or None if not found.
        """
        row = await self.get(entry_id)
        if not row:
            return None

        # Snapshot the current version before modifying
        await self._snapshot_version(row)

        # Apply updates
        for field in ("name", "description", "steps", "trigger", "success_rate", "metadata_extra"):
            if field in updates:
                setattr(row, field, updates[field])

        # Increment version
        row.version += 1
        await self._session.flush()
        return row

    async def rollback(self, entry_id: str, target_version: int | None = None) -> ProceduralRow | None:
        """Rollback a procedure to a previous version.

        If target_version is None, rolls back to the immediately previous version.
        Returns the restored row or None if not found.
        """
        row = await self.get(entry_id)
        if not row:
            return None

        if target_version is None:
            target_version = row.version - 1
        if target_version < 1:
            return None

        # Find the version snapshot
        stmt = (
            select(ProceduralVersionRow)
            .where(
                and_(
                    ProceduralVersionRow.procedure_id == entry_id,
                    ProceduralVersionRow.version == target_version,
                )
            )
            .order_by(ProceduralVersionRow.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        snapshot = result.scalar_one_or_none()
        if not snapshot:
            return None

        # Snapshot current state before rollback
        await self._snapshot_version(row)

        # Restore from snapshot
        row.name = snapshot.name
        row.description = snapshot.description
        row.steps = snapshot.steps
        row.trigger = snapshot.trigger
        row.success_rate = snapshot.success_rate
        row.metadata_extra = snapshot.metadata_extra
        row.version += 1  # increment version (rollback is a new version)
        await self._session.flush()
        return row

    async def get_version_history(self, entry_id: str) -> list[dict]:
        """Get the complete version history for a procedure."""
        stmt = (
            select(ProceduralVersionRow)
            .where(ProceduralVersionRow.procedure_id == entry_id)
            .order_by(ProceduralVersionRow.version.asc())
        )
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        return [
            {
                "version": r.version,
                "name": r.name,
                "description": r.description,
                "steps": r.steps,
                "trigger": r.trigger,
                "success_rate": r.success_rate,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    async def _snapshot_version(self, row: ProceduralRow) -> None:
        """Save a version snapshot of the current procedural row state."""
        snapshot = ProceduralVersionRow(
            procedure_id=row.id,
            version=row.version,
            name=row.name,
            description=row.description,
            steps=row.steps,
            trigger=row.trigger,
            success_rate=row.success_rate,
            metadata_extra=row.metadata_extra,
        )
        self._session.add(snapshot)
        await self._session.flush()

    async def get(self, entry_id: str) -> ProceduralRow | None:
        stmt = select(ProceduralRow).where(ProceduralRow.id == entry_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self) -> list[ProceduralRow]:
        stmt = select(ProceduralRow).order_by(ProceduralRow.name)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def find_by_trigger(self, trigger_text: str) -> list[ProceduralRow]:
        stmt = select(ProceduralRow).order_by(ProceduralRow.success_rate.desc())
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        trigger_lower = trigger_text.lower()
        return [r for r in rows if trigger_lower in r.trigger.lower() or trigger_lower in r.name.lower()]


class PolicyStore:
    """Manages permission / safety / privacy rules."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, entry: PolicyEntry) -> str:
        row = PolicyRow(
            id=str(entry.id),
            name=entry.name,
            rule=entry.rule,
            effect=entry.effect,
            priority=entry.priority,
            scope=entry.scope,
            active=entry.active,
        )
        self._session.add(row)
        await self._session.flush()
        return str(entry.id)

    async def get(self, entry_id: str) -> PolicyRow | None:
        stmt = select(PolicyRow).where(PolicyRow.id == entry_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_policies(self, scope: str | None = None) -> list[PolicyRow]:
        conditions = [PolicyRow.active == True]  # noqa: E712
        if scope:
            conditions.append(PolicyRow.scope == scope)
        stmt = (
            select(PolicyRow)
            .where(and_(*conditions))
            .order_by(PolicyRow.priority.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def check_action(self, action: str, resource: str = "") -> dict:
        """Check if an action is allowed by active policies. Returns verdict."""
        policies = await self.get_active_policies()
        action_lower = action.lower()
        resource_lower = resource.lower()
        for p in policies:
            rule_words = set(p.rule.lower().split())
            if action_lower in rule_words or resource_lower in rule_words:
                return {
                    "allowed": p.effect == "allow",
                    "effect": p.effect,
                    "policy_id": p.id,
                    "policy_name": p.name,
                    "reason": p.rule,
                }
        # Default: allow if no matching policy
        return {"allowed": True, "effect": "allow", "policy_id": None, "reason": "no_matching_policy"}
