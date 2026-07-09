"""PS4/PS5 controller teleop — pygame.joystick replaces harpcontrol's evdev.

The original (ps4control.py + rmd2.py) latched a speed on each D-pad press
until the next command. Here it's **hold-to-drive**: the D-pad state is
polled ~50 Hz and re-sent every tick, so releasing the pad stops the robot
and the base-motor deadman timeout is continuously fed while driving —
if this loop dies, the motors zero themselves within the deadman window.

Buttons (same roles as ps4control.py): Cross = stop, Square = speed up,
Circle = speed down. SDL exposes DualShock/DualSense pads in one of two
joystick layouts, picked per controller by whether it reports a hat:

- **HIDAPI** (pygame 2's default driver for Sony pads — the DS4 on this
  machine): no hat; the D-pad is buttons 11-14, Cross=0, Circle=1, Square=2.
- **Classic DirectInput** (older driver / some BT paths): D-pad is hat 0,
  Square=0, Cross=1, Circle=2.

If presses land on the wrong action, run
`python -m harp.motion --test-controller` and adjust the layouts below.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

logger = logging.getLogger(__name__)

SPEED_STEP = 100
MAX_SPEED = 2000
MIN_SPEED = 0

_POLL_INTERVAL = 0.02  # 50 Hz — well inside the motors' deadman window


@dataclass(frozen=True)
class _Layout:
    cross: int
    circle: int
    square: int
    # D-pad button indices when the pad exposes no hat (HIDAPI layout).
    dpad_up: int | None = None
    dpad_down: int | None = None
    dpad_left: int | None = None
    dpad_right: int | None = None


_DINPUT = _Layout(cross=1, circle=2, square=0)
_HIDAPI = _Layout(
    cross=0, circle=1, square=2, dpad_up=11, dpad_down=12, dpad_left=13, dpad_right=14
)


def _layout_for(joystick) -> _Layout:
    return _DINPUT if joystick.get_numhats() > 0 else _HIDAPI


def _dpad_state(joystick, layout: _Layout) -> tuple[int, int]:
    """Current D-pad state as a hat-style (x, y) regardless of layout."""
    if joystick.get_numhats() > 0:
        return joystick.get_hat(0)
    x = int(joystick.get_button(layout.dpad_right)) - int(
        joystick.get_button(layout.dpad_left)
    )
    y = int(joystick.get_button(layout.dpad_up)) - int(
        joystick.get_button(layout.dpad_down)
    )
    return x, y


def drive_speeds(
    hat_x: int, hat_y: int, base_speed: int, turn_speed: int
) -> tuple[int, int]:
    """D-pad state → (left_rpm, right_rpm).

    Signs are preserved exactly from rmd2.py's hardware-proven mapping
    (UP = rmd2 'w', DOWN = 'x', LEFT = 'a', RIGHT = 'd' — note the original's
    Forward/Reverse print labels disagreed with each other; the bytes on the
    wire are what mattered and are unchanged here). Forward/back wins if the
    pad reports a diagonal.
    """
    if hat_y == 1:
        return base_speed, -base_speed
    if hat_y == -1:
        return -base_speed, base_speed
    if hat_x == -1:
        return 0, -turn_speed
    if hat_x == 1:
        return turn_speed, 0
    return 0, 0


class PS5Teleop:
    """Polls the controller and feeds a `send(left_rpm, right_rpm)` callable."""

    def __init__(
        self,
        send: Callable[[int, int], None],
        base_speed: int = 500,
        turn_speed: int = 500,
    ) -> None:
        self._send = send
        self.base_speed = base_speed
        self.turn_speed = turn_speed

    def run(self, stop_event: threading.Event) -> None:
        """Blocking loop (run it on the main thread — pygame prefers that).

        Waits for a controller if none is plugged in yet; survives unplug/
        replug (pygame posts JOYDEVICEADDED for already-present devices too).
        """
        pygame.init()
        pygame.joystick.init()
        joystick: pygame.joystick.JoystickType | None = None
        layout = _DINPUT
        logger.info("teleop: waiting for a controller (D-pad drives, Cross stops)")

        try:
            while not stop_event.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.JOYDEVICEADDED and joystick is None:
                        joystick = pygame.joystick.Joystick(event.device_index)
                        layout = _layout_for(joystick)
                        logger.info(
                            "teleop: controller connected: %s (%s layout)",
                            joystick.get_name(),
                            "hat/DirectInput" if joystick.get_numhats() else "HIDAPI",
                        )
                    elif event.type == pygame.JOYDEVICEREMOVED and joystick is not None:
                        if event.instance_id == joystick.get_instance_id():
                            joystick = None
                            self._send(0, 0)
                            logger.warning("teleop: controller disconnected — stop")
                    elif event.type == pygame.JOYBUTTONDOWN and joystick is not None:
                        self._on_button(event.button, layout)

                if joystick is not None:
                    x, y = _dpad_state(joystick, layout)
                    left, right = drive_speeds(
                        x, y, self.base_speed, self.turn_speed
                    )
                    # Sent every tick, even (0, 0) — this is what feeds the
                    # motors' deadman timeout while the teleop loop is alive.
                    self._send(left, right)

                stop_event.wait(_POLL_INTERVAL)
        finally:
            self._send(0, 0)
            pygame.quit()

    def _on_button(self, button: int, layout: _Layout) -> None:
        if button == layout.cross:
            self._send(0, 0)
            logger.info("teleop: STOP (Cross)")
        elif button == layout.square:
            self.base_speed = min(MAX_SPEED, self.base_speed + SPEED_STEP)
            logger.info("teleop: speed up → %d", self.base_speed)
        elif button == layout.circle:
            self.base_speed = max(MIN_SPEED, self.base_speed - SPEED_STEP)
            logger.info("teleop: speed down → %d", self.base_speed)


def run_test(stop_event: threading.Event | None = None) -> None:
    """Print every button/hat event so the index constants can be verified."""
    pygame.init()
    pygame.joystick.init()
    print("Press controller buttons / D-pad (Ctrl+C to quit)...")
    sticks: dict[int, pygame.joystick.JoystickType] = {}
    try:
        while stop_event is None or not stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEADDED:
                    js = pygame.joystick.Joystick(event.device_index)
                    sticks[js.get_instance_id()] = js
                    print(f"connected: {js.get_name()!r} "
                          f"(buttons={js.get_numbuttons()} hats={js.get_numhats()} → "
                          f"{'hat/DirectInput' if js.get_numhats() else 'HIDAPI'} layout)")
                elif event.type == pygame.JOYBUTTONDOWN:
                    print(f"button {event.button} DOWN")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"button {event.button} up")
                elif event.type == pygame.JOYHATMOTION:
                    print(f"hat {event.hat} = {event.value}")
            pygame.time.wait(20)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
