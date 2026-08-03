import unittest

from config_builder import build_config


class ConfigBuilderTest(unittest.TestCase):
    def test_uses_only_authoritative_archive_facts(self):
        self.assertEqual(
            build_config(),
            {"mode": "strict", "retry_limit": 4, "timeout_ms": 2750},
        )


if __name__ == "__main__":
    unittest.main()
