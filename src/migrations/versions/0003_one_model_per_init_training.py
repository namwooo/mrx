"""one line model per init training

Revision ID: 0003
Revises: 0002
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("uq_models_init_training", "models", ["init_training_id"], unique=True)


def downgrade() -> None:
    op.drop_index("uq_models_init_training", table_name="models")
