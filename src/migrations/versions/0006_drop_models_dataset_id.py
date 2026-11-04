"""drop models.dataset_id

Dropped in place instead of rewriting the table, so uq_models_applied_scope stays.

Revision ID: 0006
Revises: 0005
"""
from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("models", "dataset_id")


def downgrade() -> None:
    op.add_column("models", sa.Column("dataset_id", sa.Integer(), nullable=True))
