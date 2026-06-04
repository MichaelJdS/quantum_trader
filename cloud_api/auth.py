"""
cloud_api/auth.py — Autenticação por Bearer Token para a Cloud API
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


async def verify_token(authorization: str = Header(default="")) -> str:
    """
    Dependência FastAPI que valida o Bearer Token.
    Token configurado via variável de ambiente API_TOKEN.
    Se API_TOKEN não estiver definida, aceita qualquer token (modo dev).
    """
    api_token = os.getenv("API_TOKEN", "")

    if not api_token:
        # Modo desenvolvimento — sem autenticação
        return "dev"

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header inválido. Use: Bearer <token>",
        )

    token = authorization.removeprefix("Bearer ").strip()
    if token != api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido.",
        )
    return token
