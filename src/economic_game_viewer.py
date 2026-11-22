"""Read-only future Game viewer preserving prior and older recorded history.

Launching this viewer never starts/stops a model or a Game. The operator must
retarget the existing UI only after the prior Game is closed and the new
collector has created its display metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy

from . import economic_identity_registry as registry
from .external_run_viewer import ExternalRunViewer
from .read_only_game_history import CombinedGameHistory

ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = ROOT / '.local/ongoing_curriculum_iteration2_48h_20261114/presentation/read_only_comparison_viewer.py'
PREVIOUS_ROOT = ROOT / '.local/_in-progress/sp-20261116-10-bdacaf88-c01'
PREVIOUS_PREFIX_BATCH = ROOT / '.local/_in-progress/sp-20261116-10-bdacaf88-compact-bounded/batch_01'
OLDER_ROOT = ROOT / '.local/ongoing_curriculum_iteration2_48h_20261114/actual_comparison_v1'


def local_directory(path):
    path = Path(path)
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to((ROOT / '.local').resolve()) or not resolved.is_dir():
        raise ValueError('Read-only viewer input must be an existing local evidence directory')
    return resolved


class EmptyHistory:
    def __init__(self, warning=None):
        self.warning = warning

    def snapshot(self):
        return {'runs': [], 'read_only': True, 'viewer_warning': self.warning}


def recorded_history(current, previous_root=None, *, previous_prefix_batch=None, older_root=None):
    """Preserve c01 plus older recorded episodes; queued entries are excluded."""
    previous = ExternalRunViewer(local_directory(previous_root)) if previous_root else EmptyHistory('Previous Game history not configured')
    if previous_prefix_batch is not None:
        if not isinstance(previous, ExternalRunViewer):
            raise ValueError('A prior prefix needs its previous Game root')
        prefix = local_directory(previous_prefix_batch)
        previous._manifest(prefix / 'MANIFEST.json')
        previous._tail(prefix / 'EVENTS.jsonl')
    older = ExternalRunViewer(local_directory(older_root)) if older_root else EmptyHistory('Older history not configured')
    return CombinedGameHistory(current, CombinedGameHistory(previous, older))


def build_viewers(folder, *, previous_root=None, previous_prefix_batch=None, older_root=None):
    """Construct read-only viewers from already recorded metadata only."""
    folder = local_directory(folder)
    plan = json.loads((folder / 'PLAN.json').read_bytes())
    registry.register_candidate(plan['economic_identity_candidate'])
    if not PRESENTATION.is_file():
        raise ValueError('The existing read-only presentation is unavailable')
    namespace = runpy.run_path(str(PRESENTATION), run_name='read_only_economic_viewer')
    arena = namespace['ComparisonViewer'](folder, folder/'DISPLAY_CATALOG.json', folder/'DISPLAY_METADATA.json')
    return namespace, arena, recorded_history(arena, previous_root,
        previous_prefix_batch=previous_prefix_batch, older_root=older_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--previous-root', type=Path, default=PREVIOUS_ROOT)
    parser.add_argument('--previous-prefix-batch', type=Path, default=PREVIOUS_PREFIX_BATCH)
    parser.add_argument('--older-root', type=Path, default=OLDER_ROOT)
    parser.add_argument('--port', type=int, default=7590)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('A valid local port is required')
    namespace, arena, history = build_viewers(args.evidence_root, previous_root=args.previous_root,
        previous_prefix_batch=args.previous_prefix_batch, older_root=args.older_root)
    namespace['serve_read_only'](arena, args.port, history)


if __name__ == '__main__': main()
