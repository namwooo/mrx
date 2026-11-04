"""integer id for models and the init_trainings table

models is rewritten, because SQLite cannot change a primary key in place. Export jobs keep pointing at their model.

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

SHARED_COLUMNS = (
    "type, model_name, model_version, task, line, weights_path, metrics, applied, applied_at, apply_comment, created_at"
)


def _models_table(id_column: sa.Column, source_id_type: sa.types.TypeEngine, **options) -> None:
    op.create_table(
        "models",
        id_column,
        sa.Column("type", sa.String(length=16), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("task", sa.String(length=64), nullable=False),
        sa.Column("line", sa.String(length=64), nullable=True),
        sa.Column("weights_path", sa.Text(), nullable=True),
        sa.Column("training_id", source_id_type, nullable=True),
        sa.Column("init_training_id", source_id_type, nullable=True),
        sa.Column("dataset_id", source_id_type, nullable=True),
        sa.Column("metrics", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("applied", sa.Boolean(), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("apply_comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint(id_column.name),
        sa.UniqueConstraint("model_name", "model_version", name="uq_models_name_version"),
        **options,
    )


def _applied_scope_index() -> None:
    op.create_index(
        "uq_models_applied_scope",
        "models",
        [sa.text("coalesce(line, '')"), "task"],
        unique=True,
        sqlite_where=sa.text("applied = 1"),
    )


def _point_export_jobs_at_new_ids(new_id: str, old_id: str) -> None:
    op.execute(
        f"""
        UPDATE jobs SET payload = json_set(payload, '$.model_id', (
            SELECT new.{new_id} FROM models AS new
            JOIN models_old AS old USING (model_name, model_version)
            WHERE old.{old_id} = json_extract(jobs.payload, '$.model_id')
        ))
        WHERE job_type = 'model_export'
        """
    )


def upgrade() -> None:
    op.drop_index("uq_models_applied_scope", table_name="models")
    op.rename_table("models", "models_old")
    _models_table(sa.Column("id", sa.Integer(), nullable=False), sa.Integer(), sqlite_autoincrement=True)
    op.execute(
        f"INSERT INTO models ({SHARED_COLUMNS}) SELECT {SHARED_COLUMNS} FROM models_old ORDER BY created_at, model_id"
    )
    _point_export_jobs_at_new_ids(new_id="id", old_id="model_id")
    op.drop_table("models_old")
    _applied_scope_index()

    op.create_table(
        "init_trainings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("label_path", sa.Text(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("epochs_done", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"]),
        sqlite_autoincrement=True,
    )


def downgrade() -> None:
    op.drop_table("init_trainings")

    op.drop_index("uq_models_applied_scope", table_name="models")
    op.rename_table("models", "models_old")
    _models_table(sa.Column("model_id", sa.String(length=32), nullable=False), sa.String(length=32))
    op.execute(
        f"INSERT INTO models (model_id, {SHARED_COLUMNS}) "
        f"SELECT 'mdl_' || id, {SHARED_COLUMNS} FROM models_old"
    )
    _point_export_jobs_at_new_ids(new_id="model_id", old_id="id")
    op.drop_table("models_old")
    _applied_scope_index()
