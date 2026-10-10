"""Round 15: consistency_findings, consistency_fixes and the eval kinds consistency, cost, report.

Widens ``eval_runs.kind`` by ``consistency`` (the weekly audit of the bot against itself,
R-EVAL-004), ``cost`` (the month's cost against its ceiling, R-EVAL-007) and ``report`` (the
written summary of all evaluations, R-EVAL-008); the column stays 16 wide (the longest kind is
``proactive_audit``).  A CHECK constraint cannot be changed in SQLite, so the table is made again;
its children (``eval_items``, which reference it with ``ON DELETE CASCADE``) would be deleted with
the old table while foreign keys are on, so their rows are kept aside and put back.  The kinds the
constraint holds are read from the table itself, so a kind another round added is kept, up and down
(as rounds 10 and 12 do).

The two new tables are made after the rebuild and dropped before it on the way down: they reference
``eval_runs`` too, and a rebuild with their rows in place would delete them.

Revision ID: 0017_eval_kinds
Revises: 0016_proactive_tables
Create Date: 2026-10-10
"""

import re
from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0017_eval_kinds"
down_revision: str | None = "0016_proactive_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NEW_KINDS = ("consistency", "cost", "report")
KIND_WIDTH = 16
_KIND_LIST = re.compile(r"kind\s+IN\s*\(([^)]*)\)", re.IGNORECASE)
FINDING_STATUSES = ("proposed", "confirmed", "rejected")
FINDING_SEVERITIES = ("obvious", "minor")
FIX_ACTIONS = ("invalidate_fact", "rewrite_fact", "invalidate_lifeline", "rewrite_lifeline")
FIX_STATUSES = ("proposed", "applied", "declined", "stale")
RUN_COLUMNS = (
    "id, kind, status, mode, milestone, verdict, backends, batch_ids, params, summary, "
    "finished_at, created_at, updated_at"
)


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def _in_list(column: str, values: Sequence[str]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _eval_kinds() -> list[str]:
    """The kinds ``eval_runs`` accepts now, read from its table definition."""
    row = (
        op.get_bind()
        .exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'eval_runs'"
        )
        .fetchone()
    )
    found = _KIND_LIST.search(str(row[0])) if row else None
    if found is None:
        raise RuntimeError("cannot read the kinds of eval_runs from its table definition")
    return re.findall(r"'([^']+)'", found.group(1))


def _make_eval_runs(name: str, kinds: Sequence[str]) -> None:
    op.create_table(
        name,
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("kind", sa.String(length=KIND_WIDTH), nullable=False),
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


def _rebuild_eval_runs(kinds: Sequence[str]) -> None:
    """Make ``eval_runs`` again with another ``kind`` constraint, keeping rows and children."""
    bind = op.get_bind()
    bind.exec_driver_sql("CREATE TEMP TABLE _kept_eval_items AS SELECT * FROM eval_items")
    _make_eval_runs("_eval_runs_next", kinds)
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
    kinds = _eval_kinds()
    kinds.extend(kind for kind in NEW_KINDS if kind not in kinds)
    _rebuild_eval_runs(kinds)
    op.create_table(
        "consistency_findings",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("run_id", sa.String(length=26), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("model_severity", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("severity", sa.String(length=8), nullable=True),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("inherited_from", sa.String(length=26), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            _in_list("status", FINDING_STATUSES), name=op.f("ck_consistency_findings_status")
        ),
        sa.CheckConstraint(
            _in_list("model_severity", FINDING_SEVERITIES),
            name=op.f("ck_consistency_findings_model_severity"),
        ),
        sa.CheckConstraint(
            "severity IS NULL OR " + _in_list("severity", FINDING_SEVERITIES),
            name=op.f("ck_consistency_findings_severity"),
        ),
        sa.CheckConstraint(
            "(status = 'confirmed') = (severity IS NOT NULL)",
            name=op.f("ck_consistency_findings_decided_severity"),
        ),
        sa.CheckConstraint("seq >= 0", name=op.f("ck_consistency_findings_seq")),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["eval_runs.id"],
            name=op.f("fk_consistency_findings_run_id_eval_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_consistency_findings")),
        sa.UniqueConstraint("run_id", "seq", name="uq_consistency_findings_run_seq"),
    )
    op.create_index(
        "ix_consistency_findings_run_status",
        "consistency_findings",
        ["run_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_consistency_findings_fingerprint", "consistency_findings", ["fingerprint"], unique=False
    )
    op.create_table(
        "consistency_fixes",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("finding_id", sa.String(length=26), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=20), nullable=False),
        sa.Column("target_id", sa.String(length=26), nullable=False),
        sa.Column("status", sa.String(length=8), nullable=False),
        sa.Column("new_id", sa.String(length=26), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("applied_at", sa.DateTime(), nullable=True),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            _in_list("action", FIX_ACTIONS), name=op.f("ck_consistency_fixes_action")
        ),
        sa.CheckConstraint(
            _in_list("status", FIX_STATUSES), name=op.f("ck_consistency_fixes_status")
        ),
        sa.CheckConstraint(
            "(status = 'applied') = (applied_at IS NOT NULL)",
            name=op.f("ck_consistency_fixes_applied_at"),
        ),
        sa.CheckConstraint("seq >= 0", name=op.f("ck_consistency_fixes_seq")),
        sa.ForeignKeyConstraint(
            ["finding_id"],
            ["consistency_findings.id"],
            name=op.f("fk_consistency_fixes_finding_id_consistency_findings"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_consistency_fixes")),
        sa.UniqueConstraint("finding_id", "seq", name="uq_consistency_fixes_finding_seq"),
    )
    op.create_index(
        "ix_consistency_fixes_finding_status",
        "consistency_fixes",
        ["finding_id", "status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_consistency_fixes_finding_status", table_name="consistency_fixes")
    op.drop_table("consistency_fixes")
    op.drop_index("ix_consistency_findings_fingerprint", table_name="consistency_findings")
    op.drop_index("ix_consistency_findings_run_status", table_name="consistency_findings")
    op.drop_table("consistency_findings")
    kinds_listed = ", ".join(repr(kind) for kind in NEW_KINDS)
    op.execute(f"DELETE FROM eval_runs WHERE kind IN ({kinds_listed})")
    _rebuild_eval_runs([kind for kind in _eval_kinds() if kind not in NEW_KINDS])
