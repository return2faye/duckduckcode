from __future__ import annotations

import unittest

from duckduckcode.config import Config
from duckduckcode.core.context import ReasoningConfig


class ConfigTest(unittest.TestCase):
    def test_loads_defaults_and_environment(self) -> None:
        config = Config.from_env({"OPENAI_API_KEY": "key"})

        self.assertEqual(config.openai_api_key, "key")
        self.assertEqual(config.openai_model, "o4-mini")
        self.assertEqual(config.reasoning, ReasoningConfig("low"))
        self.assertEqual(config.context_window_tokens, 200_000)
        self.assertFalse(config.langsmith_tracing)
        self.assertIsNone(config.langsmith_api_key)
        self.assertEqual(config.langsmith_project, "duckduckcode")
        self.assertEqual(config.openai_judge_model, "gpt-5.6-terra")

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
