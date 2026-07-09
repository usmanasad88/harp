"""Safety + protocol regressions for harp/motion.

These guard the things that fail *silently* on real hardware: a wrong byte in
an RMD-X8 frame just doesn't move the wheel, a flipped teleop sign drives the
robot the wrong way, and a broken deadman means a hung teleop thread leaves
the robot driving. Golden byte values were computed with the exact algorithm
harpcontrol/gender/rmd2.py ran on the real robot.
"""

from harp.motion.base_motors import BaseMotors, make_rmdx8_speed_command
from harp.motion.gimbal import Gimbal
from harp.motion.teleop_ps5 import (
    _DINPUT,
    _HIDAPI,
    _dpad_state,
    _layout_for,
    drive_speeds,
)


class FakePort:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data) -> None:
        self.writes.append(bytes(data))

    def close(self) -> None:
        self.closed = True


ZERO_FRAME = bytes.fromhex("3EA20104E5 00000000 00".replace(" ", ""))
FWD_500_FRAME = bytes.fromhex("3EA20104E5 50C30000 13".replace(" ", ""))
REV_500_FRAME = bytes.fromhex("3EA20104E5 B03CFFFF EA".replace(" ", ""))


def test_rmdx8_frames_match_the_hardware_proven_bytes():
    assert make_rmdx8_speed_command(0) == ZERO_FRAME
    assert make_rmdx8_speed_command(500) == FWD_500_FRAME
    assert make_rmdx8_speed_command(-500) == REV_500_FRAME


def _make_motors(deadman=0.25):
    left, right = FakePort(), FakePort()
    ports = {"L": left, "R": right}
    motors = BaseMotors(
        "L", "R", deadman_seconds=deadman,
        serial_factory=lambda port, baud, timeout: ports[port],
    )
    return motors, left, right


def test_deadman_zeroes_motors_when_commands_stop_refreshing():
    motors, left, right = _make_motors(deadman=0.25)

    motors.command(500, -500)
    t0 = motors._refreshed_at

    motors._tick(t0 + 0.1)  # fresh → drives
    assert left.writes[-1] == FWD_500_FRAME
    assert right.writes[-1] == REV_500_FRAME

    motors._tick(t0 + 0.3)  # stale → zeroed
    assert left.writes[-1] == ZERO_FRAME
    assert right.writes[-1] == ZERO_FRAME


def test_motors_never_commanded_stay_zeroed():
    motors, left, right = _make_motors()
    motors._tick(12345.0)
    assert left.writes[-1] == ZERO_FRAME
    assert right.writes[-1] == ZERO_FRAME


def test_stop_zeroes_both_motors_and_closes_ports():
    motors, left, right = _make_motors()
    motors.command(500, 500)
    motors.stop()
    assert left.writes[-1] == ZERO_FRAME
    assert right.writes[-1] == ZERO_FRAME
    assert left.closed and right.closed


def test_drive_speeds_preserves_the_proven_sign_convention():
    # Signs from rmd2.py as physically driven: UP='w', DOWN='x', LEFT='a', RIGHT='d'.
    assert drive_speeds(0, 1, 500, 400) == (500, -500)
    assert drive_speeds(0, -1, 500, 400) == (-500, 500)
    assert drive_speeds(-1, 0, 500, 400) == (0, -400)
    assert drive_speeds(1, 0, 500, 400) == (400, 0)
    assert drive_speeds(0, 0, 500, 400) == (0, 0)
    # Diagonal: forward/back wins over turning.
    assert drive_speeds(1, 1, 500, 400) == (500, -500)


class FakeJoystick:
    def __init__(self, hats: int = 0, down: set[int] | None = None):
        self.hats = hats
        self.down = down or set()
        self.hat = (0, 0)

    def get_numhats(self):
        return self.hats

    def get_hat(self, i):
        return self.hat

    def get_button(self, i):
        return 1 if i in self.down else 0


def test_dpad_works_on_hidapi_pads_that_expose_no_hat():
    # The real DS4 on this machine reports 16 buttons / 0 hats (SDL HIDAPI):
    # the D-pad is buttons 11-14. Polling hat 0 there reads (0,0) forever —
    # teleop silently dead. This locks in the button-D-pad path.
    js = FakeJoystick(hats=0)
    assert _layout_for(js) is _HIDAPI
    for button, state in ((11, (0, 1)), (12, (0, -1)), (13, (-1, 0)), (14, (1, 0))):
        js.down = {button}
        assert _dpad_state(js, _HIDAPI) == state
    js.down = set()
    assert _dpad_state(js, _HIDAPI) == (0, 0)


def test_dpad_uses_the_hat_on_directinput_pads():
    js = FakeJoystick(hats=1)
    assert _layout_for(js) is _DINPUT
    js.hat = (1, -1)
    assert _dpad_state(js, _DINPUT) == (1, -1)


def test_gimbal_sends_esp32_ascii_with_the_pitch_mount_bias():
    port = FakePort()
    gimbal = Gimbal("COMX", serial_factory=lambda p, baud, timeout: port)
    gimbal.track(320, 240)  # dead-center face → hold rest, pitch biased +15
    assert port.writes == [b"Y55P55\n"]
