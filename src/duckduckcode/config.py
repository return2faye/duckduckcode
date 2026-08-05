from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Mapping
from urllib.parse import urlsplit

from dotenv import load_dotenv

from .core.context import ReasoningConfig


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str


@dataclass(frozen=True)
class Config:
    openai_api_key: str | None
    openai_model: str = "o4-mini"
    reasoning: ReasoningConfig = ReasoningConfig()
    context_window_tokens: int = 200_000
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "duckduckcode"
    openai_judge_model: str = "gpt-5.6-terra"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    agent: ModelConfig = ModelConfig("openai", "o4-mini")
    subagent: ModelConfig = ModelConfig("openai", "o4-mini")
    memory: ModelConfig = ModelConfig("openai", "o4-mini")
    judge: ModelConfig = ModelConfig("openai", "gpt-5.6-terra")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        if env is None:
            load_dotenv()
            env = os.environ

        openai_model = _legacy_model(env, "OPENAI_MODEL", "o4-mini")
        openai_judge_model = _legacy_model(env, "OPENAI_JUDGE_MODEL", "gpt-5.6-terra")
        agent = _model_config(env, "AGENT", openai_model)
        subagent = _model_config(env, "SUBAGENT", openai_model)
        memory = _model_config(env, "MEMORY", openai_model)
        judge = _model_config(env, "JUDGE", openai_judge_model)

        openai_api_key = _optional(env.get("OPENAI_API_KEY"))
        deepseek_api_key = _optional(env.get("DEEPSEEK_API_KEY"))
        selected = {
            agent.provider,
            subagent.provider,
            memory.provider,
            judge.provider,
        }
        if "openai" in selected and not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        if "deepseek" in selected and not deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")

        deepseek_base_url = env.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ).strip()
        parsed_url = urlsplit(deepseek_base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise RuntimeError("DEEPSEEK_BASE_URL must be an HTTP(S) URL")

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
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            reasoning=ReasoningConfig(env.get("OPENAI_REASONING_EFFORT", "low")),
            context_window_tokens=context_window_tokens,
            langsmith_tracing=tracing,
            langsmith_api_key=langsmith_api_key,
            langsmith_project=env.get("LANGSMITH_PROJECT", "duckduckcode"),
            openai_judge_model=openai_judge_model,
            deepseek_api_key=deepseek_api_key,
            deepseek_base_url=deepseek_base_url,
            agent=agent,
            subagent=subagent,
            memory=memory,
            judge=judge,
        )


def _model_config(
    env: Mapping[str, str], role: str, openai_default: str
) -> ModelConfig:
    provider_name = f"{role}_PROVIDER"
    provider = env.get(provider_name, "openai").strip().lower()
    if provider not in {"openai", "deepseek"}:
        raise RuntimeError(f"{provider_name} must be openai or deepseek")
    model_name = f"{role}_MODEL"
    raw_model = env.get(model_name)
    model = (
        raw_model.strip()
        if raw_model is not None
        else openai_default if provider == "openai" else "deepseek-v4-pro"
    )
    if not model:
        raise RuntimeError(f"{model_name} must be non-empty")
    return ModelConfig(provider, model)


def _legacy_model(env: Mapping[str, str], name: str, default: str) -> str:
    model = env.get(name, default).strip()
    if not model:
        raise RuntimeError(f"{name} must be non-empty")
    return model


def _optional(value: str | None) -> str | None:
    normalized = value.strip() if value is not None else ""
    return normalized or None


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise RuntimeError("LANGSMITH_TRACING must be true or false")
