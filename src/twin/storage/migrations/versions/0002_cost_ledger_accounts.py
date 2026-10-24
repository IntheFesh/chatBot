"""Round 01: cost_ledger accounts and batch ids (R-LLM-006, R-LLM-014).

Adds ``account`` (``daily`` or ``one_time``), ``batch_id``, ``reasoning_tokens`` and
``image_count`` to ``cost_ledger`` plus the indexes the summary queries use.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("cost_ledger") as batch:
        batch.add_column(
            sa.Column("account", sa.String(length=16), nullable=False, server_default="daily")
        )
        batch.add_column(sa.Column("batch_id", sa.String(length=64), nullable=True))
        batch.add_column(
            sa.Column("reasoning_tokens", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("image_count", sa.Integer(), nullable=False, server_default="0"))
        batch.create_check_constraint(
            op.f("ck_cost_ledger_account"), "account IN ('daily', 'one_time')"
        )
        batch.create_index("ix_cost_ledger_account_at", ["account", "at"])
        batch.create_index("ix_cost_ledger_batch_id", ["batch_id"])


def downgrade() -> None:
    with op.batch_alter_table("cost_ledger") as batch:
        batch.drop_index("ix_cost_ledger_batch_id")
        batch.drop_index("ix_cost_ledger_account_at")
        batch.drop_constraint(op.f("ck_cost_ledger_account"), type_="check")
        batch.drop_column("image_count")
        batch.drop_column("reasoning_tokens")
        batch.drop_column("batch_id")
        batch.drop_column("account")
