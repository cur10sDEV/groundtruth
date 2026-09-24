import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

import app.db as db_module
from app.auth.dependencies import require_member
from app.auth.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.core.errors import AuthorizationError
from app.db import init_db
from app.main import create_app
from app.models.organization import Membership, Role
from app.models.user import User


def test_password_hash_roundtrip():
    h = hash_password("secret")
    assert verify_password("secret", h)
    assert not verify_password("wrong", h)


def test_jwt_roundtrip():
    token = create_access_token(sub="u1", org_id="o1")
    payload = decode_token(token)
    assert payload["sub"] == "u1"
    assert payload["org_id"] == "o1"


def test_decode_invalid_token_raises():
    from app.core.errors import AuthenticationError

    with pytest.raises(AuthenticationError):
        decode_token("not.a.jwt")


class _StubLimiter:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[str] = []

    async def allow(self, user_key: str, limit: int, window_seconds: int) -> bool:
        self.calls.append(user_key)
        return self.allowed


@pytest.fixture
def limiter():
    return _StubLimiter()


@pytest.fixture
async def client(monkeypatch, tmp_path, limiter):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr("app.api.routes_auth.get_limiter", lambda: limiter, raising=False)
    db_module._engine = None
    db_module._sessionmaker = None
    await init_db()
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    if db_module._engine is not None:
        await db_module._engine.dispose()
    db_module._engine = None
    db_module._sessionmaker = None


async def _signup(client: AsyncClient, email: str, password: str = "pw-secret"):
    resp = await client.post(
        "/auth/signup",
        json={"email": email, "password": password, "org_name": f"org-{email}"},
    )
    assert resp.status_code == 200
    return resp.json()


async def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def test_signup_creates_user_org_owner_membership(client: AsyncClient):
    body = await _signup(client, "alice@example.com", "pw-secret")
    token = body["token"]
    payload = decode_token(token)
    assert payload["sub"] == body["user_id"]
    assert payload["org_id"] == body["org_id"]

    async with db_module.get_session() as session:
        user = (
            (await session.execute(select(User).where(User.email == "alice@example.com")))
            .scalars()
            .one()
        )
        assert user.password_hash != "pw-secret"
        assert verify_password("pw-secret", user.password_hash)

        membership = (
            (
                await session.execute(
                    select(Membership).where(
                        Membership.user_id == user.id, Membership.org_id == body["org_id"]
                    )
                )
            )
            .scalars()
            .one()
        )
        assert membership.role == Role.OWNER


async def test_signup_duplicate_email_rejected(client: AsyncClient):
    await _signup(client, "dup@example.com")
    resp = await client.post(
        "/auth/signup",
        json={"email": "dup@example.com", "password": "pw", "org_name": "org2"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"] == "email already registered"


async def test_login_returns_token_and_rejects_bad_password(client: AsyncClient):
    created = await _signup(client, "login@example.com", "pw-secret")
    resp = await client.post(
        "/auth/login", json={"email": "login@example.com", "password": "pw-secret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert decode_token(body["token"])["sub"] == created["user_id"]
    assert body["org_id"] == created["org_id"]

    bad = await client.post("/auth/login", json={"email": "login@example.com", "password": "wrong"})
    assert bad.status_code == 401
    assert bad.json()["error"] == "invalid credentials"


async def test_login_rate_limited_returns_429(client: AsyncClient, monkeypatch):
    await _signup(client, "rl@example.com", "pw-secret")
    blocked = _StubLimiter(allowed=False)
    monkeypatch.setattr("app.api.routes_auth.get_limiter", lambda: blocked, raising=False)
    resp = await client.post(
        "/auth/login", json={"email": "rl@example.com", "password": "pw-secret"}
    )
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate limit exceeded"
    assert blocked.calls == ["login:rl@example.com"]


async def test_login_calls_rate_limiter_per_email(client: AsyncClient, limiter):
    await _signup(client, "keyed@example.com", "pw-secret")
    await client.post("/auth/login", json={"email": "keyed@example.com", "password": "pw-secret"})
    assert limiter.calls == ["login:keyed@example.com"]


async def test_me_requires_valid_token(client: AsyncClient):
    resp = await client.get("/auth/me")
    assert resp.status_code == 401

    resp = await client.get("/auth/me", headers=await _auth_headers("garbage.token"))
    assert resp.status_code == 401


async def test_me_returns_actual_role_for_token_org(client: AsyncClient):
    created = await _signup(client, "owner@example.com", "pw-secret")
    resp = await client.get("/auth/me", headers=await _auth_headers(created["token"]))
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == created["user_id"]
    assert body["org_id"] == created["org_id"]
    assert body["role"] == "owner"


async def test_invite_creates_membership_in_callers_org(client: AsyncClient):
    alice = await _signup(client, "inviter@example.com", "pw-secret")
    bob = await _signup(client, "invitee@example.com", "pw-secret")

    resp = await client.post(
        "/auth/invite",
        json={"email": "invitee@example.com", "role": "member"},
        headers=await _auth_headers(alice["token"]),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["org_id"] == alice["org_id"]

    async with db_module.get_session() as session:
        memberships = (
            (await session.execute(select(Membership).where(Membership.user_id == bob["user_id"])))
            .scalars()
            .all()
        )
        by_org = {m.org_id: m.role for m in memberships}
        assert by_org[alice["org_id"]] == Role.MEMBER
        assert by_org[bob["org_id"]] == Role.OWNER


async def test_invite_requires_auth(client: AsyncClient):
    resp = await client.post("/auth/invite", json={"email": "x@example.com", "role": "member"})
    assert resp.status_code == 401


async def test_invite_unknown_email_returns_404(client: AsyncClient):
    alice = await _signup(client, "inviter2@example.com", "pw-secret")
    resp = await client.post(
        "/auth/invite",
        json={"email": "ghost@example.com", "role": "member"},
        headers=await _auth_headers(alice["token"]),
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "user not found"


async def test_invite_duplicate_membership_rejected(client: AsyncClient):
    alice = await _signup(client, "inviter3@example.com", "pw-secret")
    await _signup(client, "dupmember@example.com", "pw-secret")
    headers = await _auth_headers(alice["token"])
    first = await client.post(
        "/auth/invite",
        json={"email": "dupmember@example.com", "role": "member"},
        headers=headers,
    )
    assert first.status_code == 200
    second = await client.post(
        "/auth/invite",
        json={"email": "dupmember@example.com", "role": "member"},
        headers=headers,
    )
    assert second.status_code == 422
    assert second.json()["error"] == "already a member"


async def _seed_membership(user_id: str, org_id: str, role: Role) -> None:
    from app.models.organization import Organization

    async with db_module.get_session() as session:
        if await session.get(Organization, org_id) is None:
            session.add(Organization(id=org_id, name=f"org-{org_id}"))
        if await session.get(User, user_id) is None:
            session.add(User(id=user_id, email=f"{user_id}@example.com", password_hash="x"))
        session.add(Membership(user_id=user_id, org_id=org_id, role=role))
        await session.commit()


async def test_require_member_owner_required_checks_membership_db(client: AsyncClient):
    await _seed_membership("u-owner", "org-1", Role.OWNER)
    await _seed_membership("u-member", "org-1", Role.MEMBER)

    dep = require_member(Role.OWNER)
    assert await dep(user={"user_id": "u-owner", "org_id": "org-1"}) == {
        "user_id": "u-owner",
        "org_id": "org-1",
    }
    with pytest.raises(AuthorizationError):
        await dep(user={"user_id": "u-member", "org_id": "org-1"})
    with pytest.raises(AuthorizationError):
        await dep(user={"user_id": "u-owner", "org_id": "org-without-membership"})


async def test_require_member_member_required_accepts_all_roles(client: AsyncClient):
    await _seed_membership("u-owner", "org-1", Role.OWNER)
    await _seed_membership("u-admin", "org-1", Role.ADMIN)
    await _seed_membership("u-member", "org-1", Role.MEMBER)

    dep = require_member(Role.MEMBER)
    for user_id in ("u-owner", "u-admin", "u-member"):
        assert await dep(user={"user_id": user_id, "org_id": "org-1"}) == {
            "user_id": user_id,
            "org_id": "org-1",
        }
    with pytest.raises(AuthorizationError):
        await dep(user={"user_id": "u-outsider", "org_id": "org-1"})


async def test_require_member_no_role_only_requires_authentication(client: AsyncClient):
    dep = require_member(None)
    assert await dep(user={"user_id": "u-anyone", "org_id": "org-anywhere"}) == {
        "user_id": "u-anyone",
        "org_id": "org-anywhere",
    }


async def test_invite_denied_without_membership_in_token_org(client: AsyncClient):
    alice = await _signup(client, "real-owner@example.com", "pw-secret")
    bob = await _signup(client, "outsider@example.com", "pw-secret")

    # validly-signed token, but bob has no membership row in alice's org
    forged = create_access_token(sub=bob["user_id"], org_id=alice["org_id"])
    resp = await client.post(
        "/auth/invite",
        json={"email": "someone@example.com", "role": "member"},
        headers=await _auth_headers(forged),
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "no membership in organization"

    # alice (OWNER of that org) is still allowed
    target = await _signup(client, "someone@example.com", "pw-secret")
    ok = await client.post(
        "/auth/invite",
        json={"email": "someone@example.com", "role": "member"},
        headers=await _auth_headers(alice["token"]),
    )
    assert ok.status_code == 200
    assert ok.json()["user_id"] == target["user_id"]


async def test_invite_denied_for_member_role_inviter(client: AsyncClient):
    # forged token: a plain MEMBER (even with a real membership row) cannot invite
    alice = await _signup(client, "owner-m@example.com", "pw-secret")
    carol = await _signup(client, "member-carol@example.com", "pw-secret")

    await client.post(
        "/auth/invite",
        json={"email": "member-carol@example.com", "role": "member"},
        headers=await _auth_headers(alice["token"]),
    )
    member_headers = await _auth_headers(
        create_access_token(sub=carol["user_id"], org_id=alice["org_id"])
    )

    resp = await client.post(
        "/auth/invite",
        json={"email": "owner-m@example.com", "role": "member"},
        headers=member_headers,
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "insufficient role"


async def test_invite_role_capped_at_inviter_rank(client: AsyncClient):
    alice = await _signup(client, "rank-owner@example.com", "pw-secret")  # OWNER
    dave = await _signup(client, "rank-dave@example.com", "pw-secret")  # becomes ADMIN
    await _signup(client, "rank-erin@example.com", "pw-secret")
    await _signup(client, "rank-frank@example.com", "pw-secret")
    grace = await _signup(client, "rank-grace@example.com", "pw-secret")

    # OWNER mints an ADMIN
    promoted = await client.post(
        "/auth/invite",
        json={"email": "rank-dave@example.com", "role": "admin"},
        headers=await _auth_headers(alice["token"]),
    )
    assert promoted.status_code == 200
    assert promoted.json()["role"] == "admin"
    # dave's signup token points at his own org (where he is OWNER); forge one
    # for alice's org so his ADMIN rank there is what authorizes the invite
    admin_headers = await _auth_headers(
        create_access_token(sub=dave["user_id"], org_id=alice["org_id"])
    )

    # ADMIN can mint MEMBER and ADMIN ...
    ok_member = await client.post(
        "/auth/invite",
        json={"email": "rank-erin@example.com", "role": "member"},
        headers=admin_headers,
    )
    assert ok_member.status_code == 200

    ok_admin = await client.post(
        "/auth/invite",
        json={"email": "rank-frank@example.com", "role": "admin"},
        headers=admin_headers,
    )
    assert ok_admin.status_code == 200

    # ... but never OWNER
    forbidden = await client.post(
        "/auth/invite",
        json={"email": "rank-grace@example.com", "role": "owner"},
        headers=admin_headers,
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["error"] == "cannot invite a role above your own"

    async with db_module.get_session() as session:
        memberships = (
            (
                await session.execute(
                    select(Membership).where(Membership.user_id == grace["user_id"])
                )
            )
            .scalars()
            .all()
        )
        assert [m.org_id for m in memberships] == [grace["org_id"]]  # only her own signup org


async def test_invite_owner_can_mint_owner(client: AsyncClient):
    alice = await _signup(client, "mint-owner@example.com", "pw-secret")
    await _signup(client, "mint-zoe@example.com", "pw-secret")

    resp = await client.post(
        "/auth/invite",
        json={"email": "mint-zoe@example.com", "role": "owner"},
        headers=await _auth_headers(alice["token"]),
    )

    assert resp.status_code == 200
    assert resp.json()["role"] == "owner"
