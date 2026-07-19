# move_around tool description

> What teaches the live model when to drive the base on its stall patrol —
> and, just as important, to stop the moment anyone asks. Loaded by
> harp/motion/tools.py; only advertised when harp.yaml `motion.enabled` is
> true. Edits apply on the next HARP restart.

Drive your wheeled base on one short patrol lap around your stall: short
forward steps with lifelike look-around pauses, a turn at each corner, and
then you stop by yourself after roughly two minutes. Call it with action
'start' when a visitor asks you to move around, patrol, drive, or show how
you move — say something brief out loud first, because you keep talking and
listening while you drive. Call it with action 'stop' the MOMENT anyone asks
you to stop, wait, or stay still — stopping is instant and always safe. Do
not start moving if someone says they are standing right in front of you, and
never pretend to move: if the tool returns an error, say you cannot move
right now.
