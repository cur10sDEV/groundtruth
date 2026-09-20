from enum import StrEnum

from sqlalchemy import Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPkMixin


class Role(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"


class Organization(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "organizations"
    name: Mapped[str] = mapped_column(String(255))


class Membership(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "memberships"
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    role: Mapped[Role] = mapped_column(Enum(Role))
