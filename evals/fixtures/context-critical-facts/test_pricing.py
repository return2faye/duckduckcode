import unittest

from pricing import shipping_cents


class ShippingTest(unittest.TestCase):
    def test_contract(self):
        self.assertEqual(shipping_cents(1), 820)
        self.assertEqual(shipping_cents(3, remote=True), 1410)
        for invalid in (0, -1, 1.5, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    shipping_cents(invalid)


if __name__ == "__main__":
    unittest.main()
