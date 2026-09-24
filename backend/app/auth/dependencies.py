from fastapi import Depends, Request
from sqlalchemy import select

from app.auth.security import decode_token
from app.core.errors import AuthenticationError, AuthorizationError
from app.db import get_session
from app.models.organization import Membership, Role

ROLE_RANK = {Role.MEMBER: 0, Role.ADMIN: 1, Role.OWNER: 2}


def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise AuthenticationError(detail="missing bearer token")
    payload = decode_token(auth.removeprefix("Bearer ").strip())
    return {"user_id": payload["sub"], "org_id": payload["org_id"]}


def require_member(role: Role | None = None):
    async def _dep(user: dict = Depends(get_current_user)) -> dict:
        if role is None:
            return user
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
        if membership is None:
            raise AuthorizationError(detail="no membership in organization")
        if ROLE_RANK[membership.role] < ROLE_RANK[role]:
            raise AuthorizationError(detail="insufficient role")
        return user

    return _dep
