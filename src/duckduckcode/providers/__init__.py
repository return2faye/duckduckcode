"""Model provider integrations."""

from ..config import Config, ModelConfig
from ..core.client import Client
from .deepseek.client import DeepSeekClient
from .openai.client import OpenAIClient


def create_client(
    config: Config,
    settings: ModelConfig,
    *,
    model: str | None = None,
) -> Client:
    common = {
        "model": model or settings.model,
        "langsmith_tracing": config.langsmith_tracing,
        "langsmith_api_key": config.langsmith_api_key,
        "langsmith_project": config.langsmith_project,
    }
    if settings.provider == "openai":
        return OpenAIClient(api_key=config.openai_api_key, **common)
    if settings.provider == "deepseek":
        return DeepSeekClient(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            **common,
        )
    raise RuntimeError(f"Unknown model provider: {settings.provider}")
