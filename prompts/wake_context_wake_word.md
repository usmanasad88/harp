# Wake context — wake word

> Authoring notes stripped before use — see `harp/config.py`
> (`load_wake_context_wake_word`). Sent into the session at open when the
> always-on listener transcribed a captured phrase and found one of
> harp.yaml's `listener.wake_words` in it — see
> `harp/listener/listener.py`. `{text}` is the transcribed phrase.

(You just woke from standby because someone said: "{text}". Respond to that naturally and offer to help.)
