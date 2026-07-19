# follow_person tool description

> What teaches the live model it can follow the recognized person in front of
> it — and that this only works for people face-ID knows, and must stop the
> moment anyone asks. Loaded by harp/motion/tools.py; only advertised when
> harp.yaml `motion.enabled` is true and the camera + face-ID are running.
> Edits apply on the next HARP restart.

Follow the person you are talking to: drive slowly toward them and keep them
in front of you as they walk, stopping on your own whenever you are close
enough and resuming when they move away. This ONLY works for a person your
face recognition knows — for anyone else it returns an error, and you should
explain you can only follow people you recognize. Call it with action 'start'
when they ask you to follow them, come with them, or come along — say
something brief out loud first, and remind them to keep a safe distance from
other people and obstacles, because you keep talking and listening while you
drive. Call it with action 'stop' the MOMENT anyone asks you to stop, wait,
or stay — stopping is instant and always safe. It also stops by itself when
you lose sight of the person for a few seconds. Never pretend to follow: if
the tool returns an error, say you cannot follow right now.
