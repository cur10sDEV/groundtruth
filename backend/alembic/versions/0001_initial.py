"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-20
"""

from alembic import op

from app.models import Base

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
