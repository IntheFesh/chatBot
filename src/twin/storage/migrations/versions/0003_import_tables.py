"""Round 03 tables: conversations, messages, media_assets, stickers, sticker_uses, import_runs.

Revision ID: 0003_import_tables
Revises: 0002
Create Date: 2026-10-09
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0003_import_tables"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MESSAGE_KINDS = (
    "'text', 'sticker', 'image', 'quote', 'voice', 'video', 'file', 'link', 'call', "
    "'transfer', 'redpacket', 'system', 'location', 'chathistory', 'unknown'"
)


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("username", sa.LargeBinary(), nullable=False),
        sa.Column("display_name", sa.LargeBinary(), nullable=True),
        sa.Column("is_group", sa.Boolean(), nullable=False),
        sa.Column("her_avatar_sha256", sa.String(length=64), nullable=True),
        sa.Column("user_avatar_sha256", sa.String(length=64), nullable=True),
        sa.Column("message_count", sa.Integer(), nullable=False),
        sa.Column("first_message_at", sa.DateTime(), nullable=True),
        sa.Column("last_message_at", sa.DateTime(), nullable=True),
        sa.Column("last_export_id", sa.String(length=128), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
    )
    op.create_table(
        "stickers",
        sa.Column("md5", sa.String(length=32), nullable=False),
        sa.Column("url", sa.LargeBinary(), nullable=True),
        sa.Column("source_path", sa.LargeBinary(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.String(length=48), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("mime", sa.String(length=32), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_attempt_at", sa.DateTime(), nullable=True),
        sa.Column("her_uses", sa.Integer(), nullable=False),
        sa.Column("user_uses", sa.Integer(), nullable=False),
        sa.Column("first_used_at", sa.DateTime(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('pending', 'available', 'md5_mismatch', 'unavailable')",
            name=op.f("ck_stickers_status"),
        ),
        sa.PrimaryKeyConstraint("md5", name=op.f("pk_stickers")),
    )
    op.create_index("ix_stickers_sha256", "stickers", ["sha256"], unique=False)
    op.create_index("ix_stickers_status", "stickers", ["status"], unique=False)
    op.create_table(
        "import_runs",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("export_id", sa.String(length=128), nullable=True),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("source_dir", sa.LargeBinary(), nullable=False),
        sa.Column("target_username", sa.LargeBinary(), nullable=False),
        sa.Column("conversation_id", sa.String(length=26), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("phase", sa.String(length=16), nullable=False),
        sa.Column("total", sa.Integer(), nullable=True),
        sa.Column("processed", sa.Integer(), nullable=False),
        sa.Column("inserted", sa.Integer(), nullable=False),
        sa.Column("duplicates", sa.Integer(), nullable=False),
        sa.Column("conflict_updated", sa.Integer(), nullable=False),
        sa.Column("conflict_kept", sa.Integer(), nullable=False),
        sa.Column("invalid", sa.Integer(), nullable=False),
        sa.Column("last_sort_seq", sa.BigInteger(), nullable=True),
        sa.Column("last_message_id", sa.String(length=128), nullable=True),
        sa.Column("speed_per_s", sa.Float(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("report_path", sa.Text(), nullable=True),
        sa.Column("stats", sa.LargeBinary(), nullable=False),
        sa.Column("hooks", sa.LargeBinary(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "phase IN ('queued', 'discover', 'messages', 'media', 'stickers', 'finalize', "
            "'hooks', 'done')",
            name=op.f("ck_import_runs_phase"),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'interrupted', 'failed', 'done', 'superseded')",
            name=op.f("ck_import_runs_status"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_import_runs_conversation_id_conversations"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_import_runs")),
    )
    op.create_index("ix_import_runs_fingerprint", "import_runs", ["fingerprint"], unique=False)
    op.create_index("ix_import_runs_status", "import_runs", ["status"], unique=False)
    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("conversation_id", sa.String(length=26), nullable=False),
        sa.Column("create_time_utc", sa.DateTime(), nullable=False),
        sa.Column("sort_seq", sa.BigInteger(), nullable=True),
        sa.Column("local_id", sa.String(length=64), nullable=True),
        sa.Column("server_id", sa.String(length=64), nullable=True),
        sa.Column("is_sent", sa.Boolean(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("render_type", sa.String(length=48), nullable=True),
        sa.Column("text", sa.LargeBinary(), nullable=True),
        sa.Column("raw", sa.LargeBinary(), nullable=False),
        sa.Column("sticker_md5", sa.String(length=32), nullable=True),
        sa.Column("media_sha256", sa.String(length=64), nullable=True),
        sa.Column("quote", sa.LargeBinary(), nullable=True),
        sa.Column("call_status", sa.String(length=16), nullable=True),
        sa.Column("call_duration_s", sa.Integer(), nullable=True),
        sa.Column("voice_seconds", sa.Integer(), nullable=True),
        sa.Column("has_transcript", sa.Boolean(), nullable=False),
        sa.Column("source_export_id", sa.String(length=128), nullable=False),
        sa.Column("source_exported_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(f"kind IN ({MESSAGE_KINDS})", name=op.f("ck_messages_kind")),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_messages_conversation_id_conversations"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
    )
    op.create_index(
        "ix_messages_conversation_time",
        "messages",
        ["conversation_id", "create_time_utc"],
        unique=False,
    )
    op.create_index("ix_messages_sent_kind", "messages", ["is_sent", "kind"], unique=False)
    op.create_index("ix_messages_sticker_md5", "messages", ["sticker_md5"], unique=False)
    op.create_table(
        "media_assets",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.String(length=26), nullable=False),
        sa.Column("message_id", sa.String(length=128), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.String(length=48), nullable=True),
        sa.Column("orig_md5", sa.String(length=64), nullable=True),
        sa.Column("file_id", sa.String(length=128), nullable=True),
        sa.Column("source_path", sa.LargeBinary(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("mime", sa.String(length=32), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("caption", sa.LargeBinary(), nullable=True),
        sa.Column("caption_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "kind IN ('image', 'video_cover', 'voice', 'avatar')",
            name=op.f("ck_media_assets_kind"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'available', 'missing', 'corrupt')",
            name=op.f("ck_media_assets_status"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_media_assets_conversation_id_conversations"),
        ),
        sa.ForeignKeyConstraint(
            ["message_id"], ["messages.id"], name=op.f("fk_media_assets_message_id_messages")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_media_assets")),
    )
    op.create_index("ix_media_assets_kind_status", "media_assets", ["kind", "status"], unique=False)
    op.create_index("ix_media_assets_message_id", "media_assets", ["message_id"], unique=False)
    op.create_index("ix_media_assets_sha256", "media_assets", ["sha256"], unique=False)
    op.create_index("ix_media_assets_status", "media_assets", ["status"], unique=False)
    op.create_table(
        "sticker_uses",
        sa.Column("message_id", sa.String(length=128), nullable=False),
        sa.Column("sticker_md5", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.String(length=26), nullable=False),
        sa.Column("by_her", sa.Boolean(), nullable=False),
        sa.Column("used_at", sa.DateTime(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_sticker_uses_conversation_id_conversations"),
        ),
        sa.ForeignKeyConstraint(
            ["message_id"], ["messages.id"], name=op.f("fk_sticker_uses_message_id_messages")
        ),
        sa.ForeignKeyConstraint(
            ["sticker_md5"], ["stickers.md5"], name=op.f("fk_sticker_uses_sticker_md5_stickers")
        ),
        sa.PrimaryKeyConstraint("message_id", name=op.f("pk_sticker_uses")),
    )
    op.create_index("ix_sticker_uses_sticker_md5", "sticker_uses", ["sticker_md5"], unique=False)
    op.create_index("ix_sticker_uses_used_at", "sticker_uses", ["used_at"], unique=False)


def downgrade() -> None:
    op.drop_table("sticker_uses")
    op.drop_table("media_assets")
    op.drop_table("messages")
    op.drop_table("import_runs")
    op.drop_table("stickers")
    op.drop_table("conversations")
