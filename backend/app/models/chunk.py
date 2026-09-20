from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPkMixin


class Chunk(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "chunks"
    doc_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    chunk_text: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int] = mapped_column(Integer, default=0)
    start_offset: Mapped[int] = mapped_column(Integer, default=0)
    end_offset: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
