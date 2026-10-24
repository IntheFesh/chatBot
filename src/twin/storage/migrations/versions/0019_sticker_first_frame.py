"""Round 17: ``stickers.first_frame_sha256`` - the still first frame of a GIF, by its digest.

The setting ``stickers.gif_send_mode: first_frame`` makes the bot send the first frame of a GIF
sticker as a PNG instead of the animated file (R-STK-007, D-696).  The PNG is stored encrypted in
the media store like every picture; the sticker row records its SHA-256 so that

* the channel's allow-list (:func:`~twin.stickers.library.sticker_file_allowed`) accepts the PNG
  as well as the original, and nothing else that is not a library sticker;
* a picture the user sends that is the PNG (forwarded back) is recognised as the same sticker;
* the label of the sticker can be found from the digest of the file that was shown.

The column is nullable (a GIF stored before this revision has no variant until the backfill
``twin stickers first-frames`` or the first send in that mode makes one) and indexed, like
``sha256``.

Revision ID: 0019_sticker_first_frame
Revises: 0018_jobs_error_detail
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019_sticker_first_frame"
down_revision: str | None = "0018_jobs_error_detail"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX = "ix_stickers_first_frame_sha256"


def upgrade() -> None:
    op.add_column("stickers", sa.Column("first_frame_sha256", sa.String(length=64), nullable=True))
    op.create_index(INDEX, "stickers", ["first_frame_sha256"], unique=False)


def downgrade() -> None:
    op.drop_index(INDEX, table_name="stickers")
    with op.batch_alter_table("stickers") as batch:
        batch.drop_column("first_frame_sha256")
