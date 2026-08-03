from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Mapping

from dotenv import load_dotenv

from .core.context import ReasoningConfig


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    openai_model: str = "o4-mini"
    reasoning: ReasoningConfig = ReasoningConfig()
    context_window_tokens: int = 200_000
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "duckduckcode"
    openai_judge_model: str = "gpt-5.6-terra"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        if env is None:
            load_dotenv()
            env = os.environ

        api_key = env.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        try:
            context_window_tokens = int(env.get("CONTEXT_WINDOW_TOKENS", "200000"))
        except ValueError as exc:
            raise RuntimeError("CONTEXT_WINDOW_TOKENS must be an integer") from exc
        if context_window_tokens <= 33_000:
            raise RuntimeError("CONTEXT_WINDOW_TOKENS must be greater than 33000")

        tracing = _boolean(env.get("LANGSMITH_TRACING", "false"))
        langsmith_api_key = env.get("LANGSMITH_API_KEY") or None
        if tracing and not langsmith_api_key:
            raise RuntimeError(
                "LANGSMITH_API_KEY is required when LANGSMITH_TRACING is true"
            )

        return cls(
            openai_api_key=api_key,
            openai_model=env.get("OPENAI_MODEL", "o4-mini"),
            reasoning=ReasoningConfig(env.get("OPENAI_REASONING_EFFORT", "low")),
            context_window_tokens=context_window_tokens,
            langsmith_tracing=tracing,
            langsmith_api_key=langsmith_api_key,
            langsmith_project=env.get("LANGSMITH_PROJECT", "duckduckcode"),
            openai_judge_model=env.get("OPENAI_JUDGE_MODEL", "gpt-5.6-terra"),
        )


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise RuntimeError("LANGSMITH_TRACING must be true or false")
