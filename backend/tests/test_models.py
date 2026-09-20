import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session, init_db
from app.models.document import Document, DocumentStatus


@pytest.fixture(autouse=True)
def _sqlite_database_url(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")


@pytest.fixture
async def session():
    await init_db()
    async with get_session() as s:
        yield s


async def test_document_round_trip(session: AsyncSession):
    doc = Document(
        id="11111111-1111-1111-1111-111111111111",
        user_id="22222222-2222-2222-2222-222222222222",
        org_id="33333333-3333-3333-3333-333333333333",
        original_filename="a.txt",
        status=DocumentStatus.PENDING,
        content_hash="abc",
        current_version=1,
    )
    session.add(doc)
    await session.commit()

    rows = (await session.execute(__import__("sqlalchemy").select(Document))).scalars().all()
    assert len(rows) == 1
    assert rows[0].status == DocumentStatus.PENDING
    assert rows[0].current_version == 1
