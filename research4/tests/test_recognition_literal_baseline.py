import unittest

from src.recognition_literal_baseline import predict


class LiteralBaselineTests(unittest.TestCase):
    def observation(self, notice):
        return [{"record_id": "example", "observation": {"clock": 12, "notice": notice}}]

    def test_canceled_notice_does_not_become_active_from_an_event_timestamp(self):
        rows = self.observation("Operations: termination of process proc-abcdef1234 at tick 14. This event was canceled; the entry is historical and no event remains scheduled.")
        result = predict(rows)[0]
        self.assertEqual(result["event_state"], "canceled")
        self.assertIsNone(result["remaining_ticks"])

    def test_multiple_identities_require_a_stronger_parser_and_are_not_silently_scored(self):
        rows = self.observation("proc-abcdef1234 monitors termination of proc-123456abcd at 14. The event is confirmed and remains scheduled.")
        with self.assertRaises(ValueError):
            predict(rows)


if __name__ == "__main__":
    unittest.main()
