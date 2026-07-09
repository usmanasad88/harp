# Wake context — push-to-talk

> Authoring notes stripped before use — see `harp/config.py`
> (`load_wake_context_push_to_talk`). Sent into the session at open when a
> press of the push-to-talk key is what woke HARP — see
> `harp/interaction/push_to_talk.py`. Push-to-talk means the person presses
> THEN speaks, so this tells the model to listen first rather than launch
> into a long welcome the person would just talk over. No placeholders.

(A person started a push-to-talk conversation: they hold a button while they speak, and are about to say something. Listen for it and answer. A very short greeting is fine, but don't give a long welcome before they have spoken.)
