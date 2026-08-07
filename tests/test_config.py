from __future__ import annotations

import unittest

from duckduckcode.config import Config, ModelConfig
from duckduckcode.core.context import ReasoningConfig


class ConfigTest(unittest.TestCase):
    def test_loads_defaults_and_environment(self) -> None:
        environment = {"OPENAI_API_KEY": "key", "MCP_TOKEN": "secret"}
        config = Config.from_env(environment)

        self.assertEqual(config.openai_api_key, "key")
        self.assertEqual(config.openai_model, "o4-mini")
        self.assertEqual(config.reasoning, ReasoningConfig("low"))
        self.assertEqual(config.context_window_tokens, 200_000)
        self.assertFalse(config.langsmith_tracing)
        self.assertIsNone(config.langsmith_api_key)
        self.assertEqual(config.langsmith_project, "duckduckcode")
        self.assertEqual(config.openai_judge_model, "gpt-5.6-terra")
        self.assertEqual(config.agent, ModelConfig("openai", "o4-mini"))
        self.assertEqual(config.subagent, ModelConfig("openai", "o4-mini"))
        self.assertEqual(config.memory, ModelConfig("openai", "o4-mini"))
        self.assertEqual(config.judge, ModelConfig("openai", "gpt-5.6-terra"))
        self.assertEqual(config.environment, environment)
        self.assertNotIn("secret", repr(config))

    def test_environment_overrides_defaults(self) -> None:
        config = Config.from_env(
            {
                "OPENAI_API_KEY": "key",
                "OPENAI_MODEL": "test-model",
                "OPENAI_REASONING_EFFORT": "medium",
                "CONTEXT_WINDOW_TOKENS": "180000",
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_API_KEY": "langsmith-key",
                "LANGSMITH_PROJECT": "test-project",
                "OPENAI_JUDGE_MODEL": "judge-model",
            }
        )

        self.assertEqual(config.openai_model, "test-model")
        self.assertEqual(config.reasoning, ReasoningConfig("medium"))
        self.assertEqual(config.context_window_tokens, 180_000)
        self.assertTrue(config.langsmith_tracing)
        self.assertEqual(config.langsmith_api_key, "langsmith-key")
        self.assertEqual(config.langsmith_project, "test-project")
        self.assertEqual(config.openai_judge_model, "judge-model")
        self.assertEqual(config.agent, ModelConfig("openai", "test-model"))
        self.assertEqual(config.judge, ModelConfig("openai", "judge-model"))

    def test_configures_each_model_role_independently(self) -> None:
        config = Config.from_env(
            {
                "AGENT_PROVIDER": "deepseek",
                "AGENT_MODEL": "deepseek-v4-pro",
                "SUBAGENT_PROVIDER": "openai",
                "SUBAGENT_MODEL": "subagent-model",
                "MEMORY_PROVIDER": "deepseek",
                "MEMORY_MODEL": "deepseek-v4-flash",
                "JUDGE_PROVIDER": "openai",
                "JUDGE_MODEL": "judge-model",
                "OPENAI_API_KEY": "openai-key",
                "DEEPSEEK_API_KEY": "deepseek-key",
                "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
            }
        )

        self.assertEqual(config.agent, ModelConfig("deepseek", "deepseek-v4-pro"))
        self.assertEqual(config.subagent, ModelConfig("openai", "subagent-model"))
        self.assertEqual(config.memory, ModelConfig("deepseek", "deepseek-v4-flash"))
        self.assertEqual(config.judge, ModelConfig("openai", "judge-model"))
        self.assertEqual(config.deepseek_api_key, "deepseek-key")
        self.assertEqual(config.deepseek_base_url, "https://deepseek.example/v1")

    def test_requires_keys_only_for_selected_providers(self) -> None:
        deepseek = Config.from_env(
            {
                "AGENT_PROVIDER": "deepseek",
                "SUBAGENT_PROVIDER": "deepseek",
                "MEMORY_PROVIDER": "deepseek",
                "JUDGE_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "deepseek-key",
            }
        )
        self.assertIsNone(deepseek.openai_api_key)
        self.assertEqual(deepseek.agent, ModelConfig("deepseek", "deepseek-v4-pro"))

        with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
            Config.from_env(
                {
                    "AGENT_PROVIDER": "deepseek",
                    "OPENAI_API_KEY": "openai-key",
                }
            )

    def test_rejects_invalid_role_configuration(self) -> None:
        cases = (
            ({"AGENT_PROVIDER": "other", "OPENAI_API_KEY": "key"}, "PROVIDER"),
            ({"AGENT_MODEL": " ", "OPENAI_API_KEY": "key"}, "AGENT_MODEL"),
            ({"OPENAI_MODEL": " ", "OPENAI_API_KEY": "key"}, "OPENAI_MODEL"),
            (
                {
                    "AGENT_PROVIDER": "deepseek",
                    "OPENAI_API_KEY": "openai-key",
                    "DEEPSEEK_API_KEY": "key",
                    "DEEPSEEK_BASE_URL": "ftp://deepseek.example",
                },
                "DEEPSEEK_BASE_URL",
            ),
        )
        for env, message in cases:
            with self.subTest(env=env):
                with self.assertRaisesRegex(RuntimeError, message):
                    Config.from_env(env)

    def test_tracing_requires_key_and_boolean_setting(self) -> None:
        for env in (
            {"OPENAI_API_KEY": "key", "LANGSMITH_TRACING": "true"},
            {"OPENAI_API_KEY": "key", "LANGSMITH_TRACING": "sometimes"},
        ):
            with self.subTest(env=env):
                with self.assertRaisesRegex(RuntimeError, "LANGSMITH"):
                    Config.from_env(env)

    def test_rejects_invalid_context_window(self) -> None:
        for value in ("large", "33000"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "CONTEXT_WINDOW_TOKENS"):
                    Config.from_env(
                        {
                            "OPENAI_API_KEY": "key",
                            "CONTEXT_WINDOW_TOKENS": value,
                        }
                    )

    def test_requires_openai_api_key(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            Config.from_env({})


if __name__ == "__main__":
    unittest.main()
