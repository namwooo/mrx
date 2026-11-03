"""jobs and models tables

Databases from before Alembic already have these tables from create_all, so each one is created only when missing.

Revision ID: 0001
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())

    if "jobs" not in existing:
        op.create_table(
            "jobs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("job_type", sa.String(length=32), nullable=False),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("result", sa.JSON(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("worker_id", sa.String(length=64), nullable=True),
            sa.Column("idempotency_key", sa.String(length=128), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("idempotency_key"),
            sqlite_autoincrement=True,
        )
        op.create_index("idx_jobs_dequeue", "jobs", ["job_type", "status", "created_at"])

    if "models" not in existing:
        op.create_table(
            "models",
            sa.Column("model_id", sa.String(length=32), nullable=False),
            sa.Column("type", sa.String(length=16), nullable=False),
            sa.Column("model_name", sa.String(length=128), nullable=False),
            sa.Column("model_version", sa.String(length=64), nullable=False),
            sa.Column("task", sa.String(length=64), nullable=False),
            sa.Column("line", sa.String(length=64), nullable=True),
            sa.Column("weights_path", sa.Text(), nullable=True),
            sa.Column("training_id", sa.String(length=32), nullable=True),
            sa.Column("init_training_id", sa.String(length=32), nullable=True),
            sa.Column("dataset_id", sa.String(length=32), nullable=True),
            sa.Column("metrics", sa.JSON(none_as_null=True), nullable=True),
            sa.Column("applied", sa.Boolean(), nullable=False),
            sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("apply_comment", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("model_id"),
            sa.UniqueConstraint("model_name", "model_version", name="uq_models_name_version"),
        )
        op.create_index(
            "uq_models_applied_scope",
            "models",
            [sa.text("coalesce(line, '')"), "task"],
            unique=True,
            sqlite_where=sa.text("applied = 1"),
        )


def downgrade() -> None:
    op.drop_table("models")
    op.drop_table("jobs")
