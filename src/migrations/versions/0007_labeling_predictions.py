"""boxes, signals and judgment of each labeled image

Revision ID: 0007
Revises: 0006
"""
from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("labeling_images", table_kwargs={"sqlite_autoincrement": True}) as batch:
        batch.add_column(sa.Column("predictions", sa.JSON(none_as_null=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("labeling_images", table_kwargs={"sqlite_autoincrement": True}) as batch:
        batch.drop_column("predictions")
