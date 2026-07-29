from __future__ import annotations

import unittest

from duckduckcode.config import Config
from duckduckcode.context import ReasoningConfig


class ConfigTest(unittest.TestCase):
    def test_loads_defaults_and_environment(self) -> None:
        config = Config.from_env({"OPENAI_API_KEY": "key"})

        self.assertEqual(config.openai_api_key, "key")
        self.assertEqual(config.openai_model, "o4-mini")
        self.assertEqual(config.reasoning, ReasoningConfig("low"))

    def test_environment_overrides_defaults(self) -> None:
        config = Config.from_env(
            {
                "OPENAI_API_KEY": "key",
                "OPENAI_MODEL": "test-model",
                "OPENAI_REASONING_EFFORT": "medium",
            }
        )

        self.assertEqual(config.openai_model, "test-model")
        self.assertEqual(config.reasoning, ReasoningConfig("medium"))

    def test_requires_openai_api_key(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            Config.from_env({})


if __name__ == "__main__":
    unittest.main()
