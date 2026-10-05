"""A queued file request is not an actual API call; cancellation is immediate."""
import tempfile
import time
import unittest
from pathlib import Path
from bridge import FileAidBridge


class BridgeCancel(unittest.TestCase):
    def test_cancellation_interrupts_wait_without_transport(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'reports').mkdir()
            (root / 'reports/CANCEL_REQUESTED.json').write_text('{}')
            bridge = FileAidBridge(root, time.monotonic() + 300)
            start = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, 'controller_requested_cancellation'):
                bridge.annotate([[]], [], 'fake_cancel_check')
            self.assertLess(time.monotonic() - start, 1)
            self.assertEqual(bridge.states, 1)
            self.assertEqual(list((root / 'annotations').glob('*.json')), [])


if __name__ == '__main__':
    unittest.main()
