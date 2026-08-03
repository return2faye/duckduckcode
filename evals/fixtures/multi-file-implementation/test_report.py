import tempfile
import unittest
from pathlib import Path

from report import render_summary


class ReportTest(unittest.TestCase):
    def test_renders_sorted_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.csv"
            path.write_text("pear,2\napple,3\npear,4\n", encoding="utf-8")
            self.assertEqual(render_summary(path), "apple: 3\npear: 6\ntotal: 9")


if __name__ == "__main__":
    unittest.main()
