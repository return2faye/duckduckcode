from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Mapping

from dotenv import load_dotenv

from .context import ReasoningConfig


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    openai_model: str = "o4-mini"
    reasoning: ReasoningConfig = ReasoningConfig()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        if env is None:
            load_dotenv()
            env = os.environ

        api_key = env.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        return cls(
            openai_api_key=api_key,
            openai_model=env.get("OPENAI_MODEL", "o4-mini"),
            reasoning=ReasoningConfig(env.get("OPENAI_REASONING_EFFORT", "low")),
        )
