from fastapi import Depends, Request

from app.auth.security import decode_token
from app.core.errors import AuthenticationError
from app.models.organization import Role


def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise AuthenticationError(detail="missing bearer token")
    payload = decode_token(auth.removeprefix("Bearer ").strip())
    return {"user_id": payload["sub"], "org_id": payload["org_id"]}


def require_member(role: Role | None = None):
    async def _dep(user: dict = Depends(get_current_user)) -> dict:
        # role check resolved against membership table in Task 4.5; v1 trusts token org_id
        return user

    return _dep
