from __future__ import annotations

import os

import streamlit as st


def get_gemini_api_key(user_input: str | None = None) -> str:
    if user_input and user_input.strip():
        return user_input.strip()

    try:
        value = st.secrets.get("GEMINI_API_KEY", "")
        if value:
            return str(value).strip()
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY", "").strip()
