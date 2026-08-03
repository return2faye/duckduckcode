import unittest

from formatter import render


class FormatterTest(unittest.TestCase):
    def test_latest_contract(self):
        self.assertEqual(render("Blue Duck", 3), "Blue Duck::3")
        self.assertEqual(render("green", 0), "green::0")


if __name__ == "__main__":
    unittest.main()
