from enum import StrEnum

from sqlalchemy import Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPkMixin


class DocumentStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    EMBEDDED = "EMBEDDED"
    FAILED = "FAILED"


class Document(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "documents"
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    original_filename: Mapped[str] = mapped_column(String(1024))
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    pending_version: Mapped[int | None] = mapped_column(Integer)
