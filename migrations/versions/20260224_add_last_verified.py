"""Add last_verified column to committed_facts

Revision ID: a1b2c3d4e5f6
Revises: None
Create Date: 2026-02-24 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add last_verified column, defaulting existing rows to their created_at value
    op.add_column(
        "committed_facts",
        sa.Column("last_verified", sa.DateTime(), nullable=True),
    )
    # Back-fill existing rows so last_verified = created_at
    op.execute("UPDATE committed_facts SET last_verified = created_at WHERE last_verified IS NULL")
    # Make column NOT NULL after back-fill
    with op.batch_alter_table("committed_facts") as batch_op:
        batch_op.alter_column(
            "last_verified",
            existing_type=sa.DateTime(),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("committed_facts") as batch_op:
        batch_op.drop_column("last_verified")
