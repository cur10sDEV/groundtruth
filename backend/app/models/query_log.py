from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPkMixin


class QueryLog(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "query_logs"
    query_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"))
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"))
    query: Mapped[str] = mapped_column(Text)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    trace_id: Mapped[str] = mapped_column(String(36))
