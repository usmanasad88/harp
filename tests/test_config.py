"""Settings-file (harp.yaml) loading: defaults, overrides, and bad input."""

from __future__ import annotations

from pathlib import Path

from harp.config import (
    DashboardSettings,
    InteractionSettings,
    ListenerSettings,
    PushToTalkSettings,
    dashboard_bind_host,
    load_settings,
)


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


def test_interaction_defaults_and_override(tmp_path: Path):
    assert load_settings(tmp_path / "nope.yaml").interaction == InteractionSettings()
    assert InteractionSettings().absence_timeout_seconds == 10.0
    f = tmp_path / "harp.yaml"
    f.write_text("interaction:\n  absence_timeout_seconds: 4\n")
    assert load_settings(f).interaction.absence_timeout_seconds == 4


def test_push_to_talk_defaults_off_and_override(tmp_path: Path):
    assert load_settings(tmp_path / "nope.yaml").push_to_talk == PushToTalkSettings()
    assert PushToTalkSettings().enabled is False
    assert PushToTalkSettings().key == "space"
    f = tmp_path / "harp.yaml"
    f.write_text("push_to_talk:\n  enabled: true\n  key: enter\n")
    ptt = load_settings(f).push_to_talk
    assert ptt.enabled is True
    assert ptt.key == "enter"


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


def test_filter_agent_defaults_and_override(tmp_path: Path):
    from harp.config import FilterAgentSettings

    assert load_settings(tmp_path / "nope.yaml").filter_agent == FilterAgentSettings()
    assert FilterAgentSettings().enabled is False
    assert FilterAgentSettings().near_field_level == 0.0
    f = tmp_path / "harp.yaml"
    f.write_text(
        "filter_agent:\n  enabled: true\n  near_field_level: 0.05\n"
        "  vad_threshold: 0.8\n  vad_silence_ms: 900\n  noise_reduction: near_field\n"
    )
    fa = load_settings(f).filter_agent
    assert fa.enabled is True
    assert fa.near_field_level == 0.05
    assert fa.vad_threshold == 0.8
    assert fa.vad_silence_ms == 900
    assert fa.noise_reduction == "near_field"


def test_filter_tuning_apply_validates_and_clamps():
    from harp.config import FilterTuning

    t = FilterTuning()
    # clamps into range
    assert t.apply("near_field_level", 5.0)["near_field_level"] == 1.0
    assert t.apply("near_field_level", -1.0)["near_field_level"] == 0.0
    assert t.apply("vad_threshold", 2.0)["vad_threshold"] == 1.0
    assert t.apply("vad_silence_ms", 50)["vad_silence_ms"] == 100     # floor
    assert t.apply("vad_silence_ms", 99999)["vad_silence_ms"] == 3000  # ceil
    # a valid enum + snapshot round-trips
    snap = t.apply("noise_reduction", "far_field")
    assert snap["noise_reduction"] == "far_field"
    assert set(snap) == {"near_field_level", "vad_threshold", "vad_silence_ms", "noise_reduction"}


def test_filter_tuning_apply_rejects_bad_input():
    import pytest

    from harp.config import FilterTuning

    t = FilterTuning()
    with pytest.raises(ValueError):
        t.apply("no_such_field", 1)
    with pytest.raises(ValueError):
        t.apply("noise_reduction", "loud")  # not one of the allowed choices
    with pytest.raises((ValueError, TypeError)):
        t.apply("vad_threshold", "not a number")
