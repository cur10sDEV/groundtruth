"""add documents.pending_version

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-22
"""

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | None = None
depends_on: str | None = None

_TABLE = "documents"
_COLUMN = "pending_version"


def _has_column(bind) -> bool:
    return _COLUMN in {c["name"] for c in sa.inspect(bind).get_columns(_TABLE)}


def upgrade() -> None:
    # 0001 is create_all-driven, so fresh databases already have this column
    # via the model metadata; the guard keeps upgrade idempotent either way.
    if not _has_column(op.get_bind()):
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.Integer(), nullable=True))


def downgrade() -> None:
    if _has_column(op.get_bind()):
        op.drop_column(_TABLE, _COLUMN)
