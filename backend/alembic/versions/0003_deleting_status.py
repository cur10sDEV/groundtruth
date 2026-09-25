"""add DELETING status + documents.failure_reason

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | None = None
depends_on: str | None = None

_TABLE = "documents"
_COLUMN = "failure_reason"


def _column_exists(insp, table: str, column: str) -> bool:
    return any(c["name"] == column for c in insp.get_columns(table))


def _enum_value_exists(conn, value: str) -> bool:
    if conn.dialect.name != "postgresql":
        return True  # sqlite fixture: create_all already has everything
    return bool(
        conn.execute(
            sa.text(
                "SELECT 1 FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid "
                "WHERE t.typname = 'documentstatus' AND e.enumlabel = :v"
            ),
            {"v": value},
        ).first()
    )


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if bind.dialect.name == "postgresql" and not _enum_value_exists(bind, "DELETING"):
        op.execute("ALTER TYPE documentstatus ADD VALUE 'DELETING'")
    if not _column_exists(insp, _TABLE, _COLUMN):
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(1024), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if _column_exists(insp, _TABLE, _COLUMN):
        op.drop_column(_TABLE, _COLUMN)
    # enum value removal is not supported in postgres; leave DELETING in place
