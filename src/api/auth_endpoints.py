"""
Auth endpoints: token login and API key management.

POST /auth/token    — exchange email+password for a JWT
POST /auth/keys     — create an API key (requires JWT)
GET  /auth/keys     — list API keys for the current user (requires JWT)
DELETE /auth/keys/{key_id} — revoke an API key (requires JWT)
POST /auth/register — create a new user (open in dev; gate behind env flag in prod)
"""

import os
from datetime import datetime, timezone
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.orm import Session

from ..echo_ridge_scoring.db_models import ApiKey, User, generate_api_key
from .auth import (
    TokenOut,
    create_access_token,
    hash_password,
    require_auth,
    verify_password,
)
from .db import get_session

router = APIRouter(prefix="/auth", tags=["auth"])

REGISTRATION_OPEN = os.environ.get("REGISTRATION_OPEN", "true").lower() == "true"


# --- Request / response schemas --------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v


class CreateKeyRequest(BaseModel):
    name: str
    rate_limit_rpm: int = 60

    @field_validator("rate_limit_rpm")
    @classmethod
    def rpm_positive(cls, v: int) -> int:
        if v < 1 or v > 10000:
            raise ValueError("rate_limit_rpm must be between 1 and 10000")
        return v


class ApiKeyOut(BaseModel):
    id: int
    name: str
    key_prefix: str
    rate_limit_rpm: int
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime]


class ApiKeyCreatedOut(ApiKeyOut):
    """Includes the raw key — shown once at creation."""
    key: str


# --- Endpoints -------------------------------------------------------------

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    body: RegisterRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Create a new user account.

    Disabled when REGISTRATION_OPEN=false (set in production).
    """
    if not REGISTRATION_OPEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Registration closed")

    existing = session.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(email=body.email, hashed_password=hash_password(body.password))
    session.add(user)
    session.commit()
    session.refresh(user)
    return {"id": user.id, "email": user.email}


@router.post("/token", response_model=TokenOut)
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Session = Depends(get_session),
) -> TokenOut:
    """Exchange email + password for a JWT access token.

    The token is valid for 24 hours (configurable via JWT_EXPIRE_HOURS).
    Pass it as `Authorization: Bearer <token>` on protected endpoints.
    """
    user: Optional[User] = session.query(User).filter(User.email == form.username).first()
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account inactive")

    token = create_access_token(user_id=user.id, email=user.email)
    return TokenOut(access_token=token)


@router.post("/keys", response_model=ApiKeyCreatedOut, status_code=status.HTTP_201_CREATED)
def create_api_key(
    body: CreateKeyRequest,
    current_user: User = Depends(require_auth),
    session: Session = Depends(get_session),
) -> ApiKeyCreatedOut:
    """Create an API key for the authenticated user.

    The full key is returned **once** in this response.  Store it securely —
    it cannot be retrieved again.  Pass it as `X-Api-Key: <key>` on requests.
    """
    raw, prefix, key_hash = generate_api_key()
    api_key = ApiKey(
        user_id=current_user.id,
        name=body.name,
        key_prefix=prefix,
        key_hash=key_hash,
        rate_limit_rpm=body.rate_limit_rpm,
    )
    session.add(api_key)
    session.commit()
    session.refresh(api_key)

    return ApiKeyCreatedOut(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        rate_limit_rpm=api_key.rate_limit_rpm,
        is_active=api_key.is_active,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        key=raw,
    )


@router.get("/keys", response_model=List[ApiKeyOut])
def list_api_keys(
    current_user: User = Depends(require_auth),
    session: Session = Depends(get_session),
) -> List[ApiKeyOut]:
    """List all API keys for the authenticated user."""
    keys = (
        session.query(ApiKey)
        .filter(ApiKey.user_id == current_user.id)
        .order_by(ApiKey.created_at.desc())
        .all()
    )
    return [
        ApiKeyOut(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            rate_limit_rpm=k.rate_limit_rpm,
            is_active=k.is_active,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
        )
        for k in keys
    ]


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_api_key(
    key_id: int,
    current_user: User = Depends(require_auth),
    session: Session = Depends(get_session),
) -> None:
    """Revoke (deactivate) an API key by ID."""
    api_key: Optional[ApiKey] = session.get(ApiKey, key_id)
    if api_key is None or api_key.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")
    api_key.is_active = False
    session.commit()
