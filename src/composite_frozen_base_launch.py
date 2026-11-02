"""Registered base-stage policy over the shared owned-child lifecycle."""
from .composite_frozen_base import MODE, BOUNDS, authorized_load_plan
from .composite_frozen_base_worker import DISK_POLICY, ARCHIVE
from .composite_frozen_launch import LaunchServices, launch as shared_launch


class BaseLaunchServices(LaunchServices):
    authorize = staticmethod(authorized_load_plan)
    mode = MODE
    bounds = BOUNDS
    disk_policy = DISK_POLICY
    worker_module = 'src.composite_frozen_base_worker'
    archive_name = ARCHIVE


def launch(root, commit, python, stage_started):
    return shared_launch(root, commit, python, stage_started, BaseLaunchServices())
