"""Settings-file (harp.yaml) loading: defaults, overrides, and bad input."""

from __future__ import annotations

from pathlib import Path

from harp.config import (
    DashboardSettings,
    InteractionSettings,
    ListenerSettings,
    MemorySettings,
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
    # a typo'd top-level key is silently ignored — this catches the repo yaml
    # and SessionLogSettings drifting apart (which would disable the log)
    assert s.session_log.enabled is True
    assert s.session_log.keep_runs > 0
    # same drift guard for the memory section (a typo would silently disable
    # long-term memory or lift the free-tier call cap)
    assert s.memory.enabled is True
    assert 0 < s.memory.calls_per_minute <= 15
    assert s.memory.model


def test_memory_defaults_and_override(tmp_path: Path):
    assert load_settings(tmp_path / "nope.yaml").memory == MemorySettings()
    f = tmp_path / "harp.yaml"
    f.write_text("memory:\n  enabled: false\n  calls_per_minute: 5\n")
    s = load_settings(f)
    assert s.memory.enabled is False and s.memory.calls_per_minute == 5
    assert s.memory.model == MemorySettings().model  # untouched key keeps default


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


def test_dashboard_open_browser_default_and_override(tmp_path: Path):
    assert DashboardSettings().open_browser is True
    f = tmp_path / "harp.yaml"
    f.write_text("dashboard:\n  open_browser: false\n")
    assert load_settings(f).dashboard.open_browser is False


def test_camera_defaults_and_override(tmp_path: Path):
    from harp.config import CameraSettings

    assert load_settings(tmp_path / "nope.yaml").camera == CameraSettings()
    assert CameraSettings().usb_webcam_index == 1
    f = tmp_path / "harp.yaml"
    f.write_text("camera:\n  backend: webcam\n  webcam_index: 2\n  usb_webcam_index: 3\n")
    cam = load_settings(f).camera
    assert cam.backend == "webcam"
    assert cam.webcam_index == 2
    assert cam.usb_webcam_index == 3


def test_camera_source_state_select_maps_to_backend_and_device():
    """This mapping is what makes the dashboard's camera dropdown able to
    tell realsense/laptop-webcam/usb-webcam apart — Camera itself only knows
    'webcam' + a device index, it has no idea which physical camera that is."""
    from harp.config import CameraSourceState

    state = CameraSourceState(webcam_index=0, usb_webcam_index=1)
    assert state.select("auto") == ("auto", 0)
    assert state.select("realsense") == ("realsense", 0)
    assert state.select("webcam") == ("webcam", 0)
    assert state.select("usb_webcam") == ("webcam", 1)
    assert state.snapshot() == {"source": "usb_webcam"}  # records the last selection


def test_camera_source_state_select_rejects_bad_input():
    import pytest

    from harp.config import CameraSourceState

    with pytest.raises(ValueError):
        CameraSourceState().select("bogus")


def test_dashboard_bind_host_maps_known_modes():
    assert dashboard_bind_host("localhost") == "127.0.0.1"
    assert dashboard_bind_host("network") == "0.0.0.0"


def test_dashboard_bind_host_falls_back_on_typo():
    assert dashboard_bind_host("nettwork") == "127.0.0.1"


def test_filter_agent_defaults_and_override(tmp_path: Path):
    from harp.config import FilterAgentSettings

    assert load_settings(tmp_path / "nope.yaml").filter_agent == FilterAgentSettings()
    assert FilterAgentSettings().enabled is False
    f = tmp_path / "harp.yaml"
    f.write_text("filter_agent:\n  enabled: true\n  response_tail_seconds: 1.5\n")
    fa = load_settings(f).filter_agent
    assert fa.enabled is True
    assert fa.response_tail_seconds == 1.5


def test_voice_tuning_defaults_and_override(tmp_path: Path):
    """voice_tuning applies to whichever agent owns the mic — the plain
    single-agent bridge, or (filter_agent.enabled) the two-agent filter."""
    from harp.config import VoiceTuningSettings

    assert load_settings(tmp_path / "nope.yaml").voice_tuning == VoiceTuningSettings()
    assert VoiceTuningSettings().near_field_level == 0.0
    f = tmp_path / "harp.yaml"
    f.write_text(
        "voice_tuning:\n  near_field_level: 0.05\n"
        "  vad_threshold: 0.8\n  vad_silence_ms: 900\n  noise_reduction: near_field\n"
    )
    vt = load_settings(f).voice_tuning
    assert vt.near_field_level == 0.05
    assert vt.vad_threshold == 0.8
    assert vt.vad_silence_ms == 900
    assert vt.noise_reduction == "near_field"


def test_voice_tuning_apply_validates_and_clamps():
    from harp.config import VoiceTuning

    t = VoiceTuning()
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


def test_voice_tuning_apply_rejects_bad_input():
    import pytest

    from harp.config import VoiceTuning

    t = VoiceTuning()
    with pytest.raises(ValueError):
        t.apply("no_such_field", 1)
    with pytest.raises(ValueError):
        t.apply("noise_reduction", "loud")  # not one of the allowed choices
    with pytest.raises((ValueError, TypeError)):
        t.apply("vad_threshold", "not a number")


def test_build_session_config_stamps_tuning_for_the_single_agent_session():
    """The plain single-agent session (not just the two-agent filter) picks up
    the dashboard's noise/VAD knobs — this is what makes the single-agent mode
    tuning panel actually do something, not just move sliders."""
    from harp.config import VoiceTuning, build_session_config

    tuning = VoiceTuning(vad_threshold=0.8, vad_silence_ms=900, noise_reduction="near_field")
    cfg = build_session_config("openai", tuning)
    assert cfg.vad_threshold == 0.8
    assert cfg.vad_silence_ms == 900
    assert cfg.noise_reduction == "near_field"


def test_build_session_config_without_tuning_leaves_vad_unset():
    from harp.config import build_session_config

    cfg = build_session_config("openai")
    assert cfg.vad_threshold is None
    assert cfg.vad_silence_ms is None
    assert cfg.noise_reduction is None
