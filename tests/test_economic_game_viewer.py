"""Synthetic read-only viewer/history composition; no service or model."""
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import Mock, patch

from src import economic_game_viewer as viewer


class StubViewer:
    def __init__(self, root): self.root = Path(root); self.rows = []
    def _manifest(self, path): self.rows.append({'id': 'retained-prefix', 'events': [1]})
    def _tail(self, path): pass
    def snapshot(self):
        return {'runs': self.rows + [{'id': self.root.name, 'events': [1]}, {'id': 'queued', 'events': []}],
                'viewer_warning': None}


class ViewerTests(unittest.TestCase):
    def test_new_current_prior_and_older_history_all_remain_selectable(self):
        with tempfile.TemporaryDirectory(dir=viewer.ROOT/'.local') as tmp:
            root = Path(tmp)
            for name in ('current', 'previous', 'prefix', 'older'): (root/name).mkdir()
            (root/'current/PLAN.json').write_text(json.dumps({'economic_identity_candidate': {'synthetic': True}}))
            presentation = root/'presentation.py'; presentation.write_text('# synthetic')
            current = StubViewer(root/'current')
            namespace = {'ComparisonViewer': Mock(return_value=current), 'serve_read_only': object()}
            with patch.object(viewer, 'PRESENTATION', presentation), patch.object(viewer, 'ExternalRunViewer', StubViewer), \
                    patch.object(viewer.registry, 'register_candidate') as register, \
                    patch.object(viewer.runpy, 'run_path', return_value=namespace):
                _, arena, history = viewer.build_viewers(root/'current', previous_root=root/'previous',
                    previous_prefix_batch=root/'prefix', older_root=root/'older')
            self.assertIs(arena, current)
            register.assert_called_once_with({'synthetic': True})
            self.assertEqual([row['id'] for row in history.snapshot()['runs']],
                             ['current', 'retained-prefix', 'previous', 'older'])
            self.assertTrue(history.snapshot()['read_only'])

    def test_inputs_outside_local_workspace_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            viewer.local_directory(tmp)

    def test_cli_preserves_prior_history_by_default(self):
        serve = Mock()
        arena, history = object(), object()
        with patch.object(sys, 'argv', ['viewer', '--evidence-root', '.local/new']), \
                patch.object(viewer, 'build_viewers', return_value=({'serve_read_only': serve}, arena, history)) as build:
            viewer.main()
        build.assert_called_once_with(Path('.local/new'), previous_root=viewer.PREVIOUS_ROOT,
                                      previous_prefix_batch=viewer.PREVIOUS_PREFIX_BATCH, older_root=viewer.OLDER_ROOT)
        serve.assert_called_once_with(arena, 7590, history)


if __name__ == '__main__': unittest.main()
