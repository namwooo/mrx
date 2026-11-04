"""labelings and labeling_images tables

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "labelings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("class_names", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("images_done", sa.Integer(), nullable=False),
        sa.Column("images_total", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"]),
        sqlite_autoincrement=True,
    )
    op.create_table(
        "labeling_images",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("labeling_id", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("labels", sa.JSON(), nullable=False),
        sa.Column("num_labels", sa.Integer(), nullable=False),
        sa.Column("suggested", sa.String(length=16), nullable=False),
        sa.Column("result", sa.String(length=16), nullable=True),
        sa.Column("reviewed_labels", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["labeling_id"], ["labelings.id"]),
        sqlite_autoincrement=True,
    )
    op.create_index("idx_labeling_images_suggested", "labeling_images", ["labeling_id", "suggested"])
    op.create_index("idx_labeling_images_result", "labeling_images", ["labeling_id", "result"])


def downgrade() -> None:
    op.drop_table("labeling_images")
    op.drop_table("labelings")
