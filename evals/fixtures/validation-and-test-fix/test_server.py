import unittest

from server import parse_port


class ParsePortTest(unittest.TestCase):
    def test_valid_port(self):
        self.assertEqual(parse_port("8080"), 8080)

    def test_invalid_port(self):
        self.assertEqual(parse_port("0"), 0)


if __name__ == "__main__":
    unittest.main()
