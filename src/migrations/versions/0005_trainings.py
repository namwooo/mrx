"""trainings and training_metrics tables, and one line model per training

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trainings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("base_model_id", sa.Integer(), nullable=True),
        sa.Column("labeling_ids", sa.JSON(), nullable=False),
        sa.Column("split", sa.JSON(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("output_dir", sa.Text(), nullable=False),
        sa.Column("epochs_done", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["base_model_id"], ["models.id"]),
        sqlite_autoincrement=True,
    )
    op.create_table(
        "training_metrics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("training_id", sa.Integer(), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["training_id"], ["trainings.id"]),
        sqlite_autoincrement=True,
    )
    op.create_index("idx_training_metrics_training", "training_metrics", ["training_id", "epoch"])
    op.create_index("uq_models_training", "models", ["training_id"], unique=True)


def downgrade() -> None:
    op.drop_index("uq_models_training", table_name="models")
    op.drop_table("training_metrics")
    op.drop_table("trainings")
