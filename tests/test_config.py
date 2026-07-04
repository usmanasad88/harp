"""Settings-file (harp.yaml) loading: defaults, overrides, and bad input."""

from __future__ import annotations

from pathlib import Path

from harp.config import DashboardSettings, ListenerSettings, dashboard_bind_host, load_settings


def test_missing_file_gives_defaults(tmp_path: Path):
    s = load_settings(tmp_path / "nope.yaml")
    assert s.listener == ListenerSettings()
    assert s.heartbeat.file == ".harp/heartbeat"
    assert s.dashboard == DashboardSettings()
    assert s.dashboard.bind == "localhost"


def test_overrides_apply_and_missing_keys_keep_defaults(tmp_path: Path):
    f = tmp_path / "harp.yaml"
    f.write_text(
        "listener:\n  wake_level: 0.5\n  wake_words: [yo]\n"
        "heartbeat:\n  interval_seconds: 2\n"
    )
    s = load_settings(f)
    assert s.listener.wake_level == 0.5
    assert s.listener.wake_words == ["yo"]
    assert s.listener.transcribe_level == ListenerSettings().transcribe_level
    assert s.heartbeat.interval_seconds == 2


def test_unknown_keys_are_ignored_not_fatal(tmp_path: Path):
    f = tmp_path / "harp.yaml"
    f.write_text("listener:\n  wake_levl: 0.9\n")  # typo'd key
    s = load_settings(f)
    assert s.listener.wake_level == ListenerSettings().wake_level


def test_repo_harp_yaml_parses():
    """The real harp.yaml in the repo must always load cleanly."""
    s = load_settings()
    assert 0.0 < s.listener.transcribe_level < s.listener.wake_level <= 1.0
    assert s.listener.wake_words
    assert s.dashboard.bind in ("localhost", "network")
    assert s.dashboard.port == 8787


def test_dashboard_bind_overrides(tmp_path: Path):
    f = tmp_path / "harp.yaml"
    f.write_text("dashboard:\n  bind: network\n  port: 9000\n")
    s = load_settings(f)
    assert s.dashboard.bind == "network"
    assert s.dashboard.port == 9000


def test_dashboard_bind_host_maps_known_modes():
    assert dashboard_bind_host("localhost") == "127.0.0.1"
    assert dashboard_bind_host("network") == "0.0.0.0"


def test_dashboard_bind_host_falls_back_on_typo():
    assert dashboard_bind_host("nettwork") == "127.0.0.1"
