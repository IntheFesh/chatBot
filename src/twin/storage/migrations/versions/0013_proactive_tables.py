"""Round 10: proactive_candidates, proactive_log, ratings and the "shared" mark of the life line.

Also widens ``eval_runs.kind`` by ``proactive_audit`` (the audit of ``twin eval proactive``).  A
CHECK constraint cannot be changed in SQLite, so the table is made again; its children
(``eval_items``, which reference it with ``ON DELETE CASCADE``) would be deleted with the old
table while foreign keys are on, so their rows are kept aside and put back.

Revision ID: 0013_proactive_tables
Revises: 0012_eval_tables
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0013_proactive_tables"
down_revision: str | None = "0012_eval_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_KINDS = ("blind", "style", "memory", "gate")
NEW_KINDS = (*OLD_KINDS, "proactive_audit")
CANDIDATE_KINDS = ("followup", "greeting", "meal", "bedtime", "silence", "share", "edge")
CANDIDATE_STATUSES = ("pending", "sent", "expired", "dropped", "declined", "superseded")
LOG_KINDS = (*CANDIDATE_KINDS, "day")
LOG_OUTCOMES = ("sent", "rejected", "declined", "failed", "expired", "dropped", "opened")
RUN_COLUMNS = (
    "id, kind, status, mode, milestone, verdict, backends, batch_ids, params, summary, "
    "finished_at, created_at, updated_at"
)


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _make_eval_runs(name: str, kinds: tuple[str, ...], kind_length: int) -> None:
    op.create_table(
        name,
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("kind", sa.String(length=kind_length), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("mode", sa.String(length=8), nullable=True),
        sa.Column("milestone", sa.String(length=2), nullable=True),
        sa.Column("verdict", sa.String(length=12), nullable=True),
        sa.Column("backends", sa.JSON(), nullable=False),
        sa.Column("batch_ids", sa.JSON(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(_in_list("kind", kinds), name=op.f("ck_eval_runs_kind")),
        sa.CheckConstraint(
            "status IN ('planned', 'running', 'done', 'cancelled', 'failed')",
            name=op.f("ck_eval_runs_status"),
        ),
        sa.CheckConstraint(
            "mode IS NULL OR mode IN ('holdout', 'live')", name=op.f("ck_eval_runs_mode")
        ),
        sa.CheckConstraint(
            "verdict IS NULL OR verdict IN ('passed', 'failed', 'insufficient')",
            name=op.f("ck_eval_runs_verdict"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_runs")),
    )


def _rebuild_eval_runs(kinds: tuple[str, ...], kind_length: int) -> None:
    """Make ``eval_runs`` again with another ``kind`` constraint, keeping rows and children."""
    bind = op.get_bind()
    bind.exec_driver_sql("CREATE TEMP TABLE _kept_eval_items AS SELECT * FROM eval_items")
    _make_eval_runs("_eval_runs_next", kinds, kind_length)
    bind.exec_driver_sql(
        f"INSERT INTO _eval_runs_next ({RUN_COLUMNS}) SELECT {RUN_COLUMNS} FROM eval_runs"
    )
    op.drop_table("eval_runs")  # with foreign keys on this deletes the children as well ...
    op.rename_table("_eval_runs_next", "eval_runs")
    op.create_index("ix_eval_runs_kind_created", "eval_runs", ["kind", "created_at"], unique=False)
    bind.exec_driver_sql("DELETE FROM eval_items")
    bind.exec_driver_sql("INSERT INTO eval_items SELECT * FROM _kept_eval_items")  # ... put back
    bind.exec_driver_sql("DROP TABLE _kept_eval_items")


def upgrade() -> None:
    op.create_table(
        "proactive_candidates",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("key", sa.String(length=48), nullable=False),
        sa.Column("kind", sa.String(length=10), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("planned_at", sa.DateTime(), nullable=False),
        sa.Column("window_end", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("status_at", sa.DateTime(), nullable=True),
        sa.Column("last_reason", sa.String(length=24), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("plan_id", sa.String(length=26), nullable=True),
        sa.Column("detail", sa.LargeBinary(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            _in_list("kind", CANDIDATE_KINDS), name=op.f("ck_proactive_candidates_kind")
        ),
        sa.CheckConstraint(
            _in_list("status", CANDIDATE_STATUSES), name=op.f("ck_proactive_candidates_status")
        ),
        sa.CheckConstraint("attempts >= 0", name=op.f("ck_proactive_candidates_attempts")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_proactive_candidates")),
        sa.UniqueConstraint("local_date", "timezone", "key", name="uq_proactive_candidates_slot"),
    )
    op.create_index(
        "ix_proactive_candidates_status_planned",
        "proactive_candidates",
        ["status", "planned_at"],
        unique=False,
    )
    op.create_table(
        "proactive_log",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("candidate_at", sa.DateTime(), nullable=False),
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("local_at", sa.String(length=16), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=10), nullable=False),
        sa.Column("outcome", sa.String(length=10), nullable=False),
        sa.Column("reason", sa.String(length=24), nullable=True),
        sa.Column("her_state", sa.String(length=12), nullable=True),
        sa.Column("chase_seq", sa.Integer(), nullable=False),
        sa.Column("bubbles_sent", sa.Integer(), nullable=False),
        sa.Column("quota_total", sa.Integer(), nullable=True),
        sa.Column("range_min", sa.Integer(), nullable=True),
        sa.Column("range_max", sa.Integer(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=True),
        sa.Column("candidate_id", sa.String(length=26), nullable=True),
        sa.Column("followup_id", sa.String(length=26), nullable=True),
        sa.Column("reply_id", sa.String(length=26), nullable=True),
        sa.Column("backend", sa.String(length=10), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("plan_reason", sa.LargeBinary(), nullable=True),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.Column("result", sa.LargeBinary(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(_in_list("kind", LOG_KINDS), name=op.f("ck_proactive_log_kind")),
        sa.CheckConstraint(
            _in_list("outcome", LOG_OUTCOMES), name=op.f("ck_proactive_log_outcome")
        ),
        sa.CheckConstraint("chase_seq >= 0", name=op.f("ck_proactive_log_chase_seq")),
        sa.CheckConstraint("bubbles_sent >= 0", name=op.f("ck_proactive_log_bubbles_sent")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_proactive_log")),
    )
    op.create_index(
        "ix_proactive_log_local_date_outcome",
        "proactive_log",
        ["local_date", "outcome"],
        unique=False,
    )
    op.create_index("ix_proactive_log_at", "proactive_log", ["at"], unique=False)
    op.create_table(
        "ratings",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=12), nullable=False),
        sa.Column("note", sa.LargeBinary(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("score >= 1 AND score <= 5", name=op.f("ck_ratings_score")),
        sa.CheckConstraint("source IN ('command')", name=op.f("ck_ratings_source")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ratings")),
    )
    op.create_index("ix_ratings_at", "ratings", ["at"], unique=False)
    op.add_column("lifeline_events", sa.Column("shared_at", sa.DateTime(), nullable=True))
    op.add_column(
        "lifeline_events", sa.Column("shared_reply_id", sa.String(length=26), nullable=True)
    )
    _rebuild_eval_runs(NEW_KINDS, 16)


def downgrade() -> None:
    op.execute("DELETE FROM eval_runs WHERE kind = 'proactive_audit'")
    _rebuild_eval_runs(OLD_KINDS, 8)
    with op.batch_alter_table("lifeline_events") as batch:
        batch.drop_column("shared_reply_id")
        batch.drop_column("shared_at")
    op.drop_index("ix_ratings_at", table_name="ratings")
    op.drop_table("ratings")
    op.drop_index("ix_proactive_log_at", table_name="proactive_log")
    op.drop_index("ix_proactive_log_local_date_outcome", table_name="proactive_log")
    op.drop_table("proactive_log")
    op.drop_index("ix_proactive_candidates_status_planned", table_name="proactive_candidates")
    op.drop_table("proactive_candidates")
