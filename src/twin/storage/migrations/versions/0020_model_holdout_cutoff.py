"""Round 18: ``model_registry.holdout_cutoff`` - the hold-out cut-off a model was trained with.

The user decided (round 18, D-885) that ``twin train export`` re-splits the hold-out by itself and
that the model registry locks the cut-off of the training set a model was made from, so that M5
draws its contexts only after it (D-886).  The column is filled when a model is registered (the
manifest of the training package names it, or the dataset version recorded at the export).

For the rows that exist it is filled from the dataset version each row is locked to:
``dataset_versions.holdout_cutoff`` is exactly the cut-off of that export.  A row whose dataset is
not recorded on this computer stays empty; such a model is not evaluated until it is registered
again with a manifest that names its cut-off.

Revision ID: 0020_model_holdout_cutoff
Revises: 0019_sticker_first_frame
Create Date: 2026-10-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020_model_holdout_cutoff"
down_revision: str | None = "0019_sticker_first_frame"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("model_registry", sa.Column("holdout_cutoff", sa.DateTime(), nullable=True))
    op.execute(
        "UPDATE model_registry SET holdout_cutoff = ("
        "SELECT dataset_versions.holdout_cutoff FROM dataset_versions "
        "WHERE dataset_versions.id = model_registry.dataset_version)"
    )


def downgrade() -> None:
    with op.batch_alter_table("model_registry") as batch:
        batch.drop_column("holdout_cutoff")
