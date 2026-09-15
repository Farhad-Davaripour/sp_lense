import json
import runpy
import socket
import sys
from pathlib import Path

from sp_lense.feature_example import main

ROOT = Path(__file__).resolve().parents[2]


def test_example(capsys):
    main()
    assert json.loads(capsys.readouterr().out)["shape"] == [2, 21]


def test_replay_with_network_connections_blocked(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Replay must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(sys, "argv", ["replay.py"])
    monkeypatch.syspath_prepend(str(ROOT / "reproduce"))
    runpy.run_path(str(ROOT / "reproduce/replay.py"), run_name="__main__")
