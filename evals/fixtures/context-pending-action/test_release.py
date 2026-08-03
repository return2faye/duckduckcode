import unittest

from release import select_release


class ReleaseTest(unittest.TestCase):
    def test_selects_stable_highest_priority_then_smallest_id(self):
        records = [
            {"id": "z", "priority": 9, "draft": True},
            {"id": "b", "priority": 5, "draft": False},
            {"id": "a", "priority": 5, "draft": False},
        ]
        self.assertEqual(select_release(records), "a")
        self.assertIsNone(select_release([{"id": "x", "priority": 1, "draft": True}]))


if __name__ == "__main__":
    unittest.main()
