from datetime import UTC, datetime, timedelta

import jwt
from passlib.context import CryptContext

from app.core.config import get_settings
from app.core.errors import AuthenticationError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_access_token(sub: str, org_id: str, expires_delta: timedelta | None = None) -> str:
    s = get_settings()
    expire = expires_delta or timedelta(minutes=s.jwt_expire_minutes)
    now = datetime.now(UTC)
    payload = {"sub": sub, "org_id": org_id, "iat": now, "exp": now + expire}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> dict:
    s = get_settings()
    try:
        return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except Exception as exc:
        raise AuthenticationError(detail="invalid or expired token") from exc
