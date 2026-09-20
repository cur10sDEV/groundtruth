from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPkMixin


class Citation(UUIDPkMixin, Base):
    __tablename__ = "citations"
    query_id: Mapped[str] = mapped_column(String(36), index=True)
    chunk_id: Mapped[str] = mapped_column(String(36), ForeignKey("chunks.id"), index=True)
    doc_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), index=True)
