"""Round 17: ``jobs.last_error_detail``, the sealed message of the exception that failed a job.

``jobs.last_error`` (plain text) holds only what is safe to read in a log or a listing: the type of
the exception, structured facts (an HTTP status, an error code) and the stack locations.  The
message of the exception - which may quote a sentence of the conversation - goes into the new
column ``last_error_detail``, sealed like every other column that can hold content (R-STO-002), and
is shown only by ``twin jobs show <id> --detail`` (D-714).

The column is nullable and has no default: the rows of before have no detail.  The plain
``last_error`` of before was written as ``<Type>: <redacted message>``; the migration cuts it back
to ``<Type>`` so that no message stays in a plain column (a migration has no key to seal it with,
and the message is only a diagnostic).  Going back down drops the column and cannot bring the cut
messages back.

Revision ID: 0018_jobs_error_detail
Revises: 0017_eval_kinds
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018_jobs_error_detail"
down_revision: str | None = "0017_eval_kinds"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("last_error_detail", sa.LargeBinary(), nullable=True))
    op.execute(
        sa.text(
            "UPDATE jobs SET last_error = substr(last_error, 1, instr(last_error, ': ') - 1) "
            "WHERE instr(last_error, ': ') > 0"
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_column("last_error_detail")
