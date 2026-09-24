from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select

from app.auth.dependencies import ROLE_RANK, get_current_user, require_member
from app.auth.security import create_access_token, hash_password, verify_password
from app.core.config import get_settings
from app.core.errors import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from app.core.redis_store import get_limiter
from app.db import get_session
from app.models.organization import Membership, Organization, Role
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])

_admin_required = require_member(Role.ADMIN)


class SignupIn(BaseModel):
    email: str
    password: str
    org_name: str


class LoginIn(BaseModel):
    email: str
    password: str


class InviteIn(BaseModel):
    email: str
    role: Role


class MeOut(BaseModel):
    user_id: str
    org_id: str
    role: str


@router.post("/signup")
async def signup(body: SignupIn) -> dict:
    async with get_session() as session:
        existing = (
            (await session.execute(select(User).where(User.email == body.email))).scalars().first()
        )
        if existing:
            raise AuthenticationError(detail="email already registered")
        org = Organization(name=body.org_name)
        session.add(org)
        await session.flush()
        user = User(email=body.email, password_hash=hash_password(body.password))
        session.add(user)
        await session.flush()
        session.add(Membership(user_id=user.id, org_id=org.id, role=Role.OWNER))
        await session.commit()
        token = create_access_token(sub=user.id, org_id=org.id)
        return {"token": token, "user_id": user.id, "org_id": org.id}


@router.post("/login")
async def login(body: LoginIn) -> dict:
    s = get_settings()
    if not await get_limiter().allow(
        f"login:{body.email}", s.rate_limit_requests, s.rate_limit_window_seconds
    ):
        raise RateLimitError(detail="rate limit exceeded")
    async with get_session() as session:
        user = (
            (await session.execute(select(User).where(User.email == body.email))).scalars().first()
        )
        if not user or not verify_password(body.password, user.password_hash):
            raise AuthenticationError(detail="invalid credentials")
        membership = (
            (await session.execute(select(Membership).where(Membership.user_id == user.id)))
            .scalars()
            .first()
        )
        org_id = membership.org_id if membership else ""
        token = create_access_token(sub=user.id, org_id=org_id)
        return {"token": token, "user_id": user.id, "org_id": org_id}


@router.post("/invite")
async def invite(body: InviteIn, user: dict = Depends(_admin_required)) -> dict:
    async with get_session() as session:
        inviter = (
            (
                await session.execute(
                    select(Membership).where(
                        Membership.user_id == user["user_id"],
                        Membership.org_id == user["org_id"],
                    )
                )
            )
            .scalars()
            .first()
        )
        if inviter is None:
            raise AuthorizationError(detail="no membership in organization")
        # requested role is capped at the inviter's own rank
        if ROLE_RANK[inviter.role] < ROLE_RANK[body.role]:
            raise AuthorizationError(detail="cannot invite a role above your own")
        target = (
            (await session.execute(select(User).where(User.email == body.email))).scalars().first()
        )
        if not target:
            raise NotFoundError(detail="user not found")
        existing = (
            (
                await session.execute(
                    select(Membership).where(
                        Membership.user_id == target.id,
                        Membership.org_id == user["org_id"],
                    )
                )
            )
            .scalars()
            .first()
        )
        if existing:
            raise ValidationError(detail="already a member")
        session.add(Membership(user_id=target.id, org_id=user["org_id"], role=body.role))
        await session.commit()
        return {
            "user_id": target.id,
            "org_id": user["org_id"],
            "role": body.role.value,
        }


@router.get("/me", response_model=MeOut)
async def me(user: dict = Depends(get_current_user)) -> MeOut:
    async with get_session() as session:
        membership = (
            (
                await session.execute(
                    select(Membership).where(
                        Membership.user_id == user["user_id"],
                        Membership.org_id == user["org_id"],
                    )
                )
            )
            .scalars()
            .first()
        )
        role = membership.role.value if membership else "member"
        return MeOut(user_id=user["user_id"], org_id=user["org_id"], role=role)
