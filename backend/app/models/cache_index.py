from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPkMixin


class CacheIndex(UUIDPkMixin, Base):
    __tablename__ = "cache_index"
    doc_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), index=True)
    cache_key: Mapped[str] = mapped_column(String(128), index=True)
