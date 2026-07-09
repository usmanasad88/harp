# Wake context — loud sound

> Authoring notes stripped before use — see `harp/config.py`
> (`load_wake_context_loud_sound`). Sent into the session at open when the
> always-on listener woke HARP because it heard a sound at or above
> `listener.wake_level` in harp.yaml — see `harp/listener/listener.py`.
> `{level}` is the measured RMS level (0.0-1.0, two decimal places).

(You just woke from standby because you heard a loud sound nearby (level {level}) — someone may be trying to get your attention. Greet them and ask how you can help.)
