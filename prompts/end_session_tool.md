# end_session — tool description

> Authoring notes stripped before the model sees this — see `harp/config.py`
> (`load_end_session_description`, shared `load_prompt`). This text becomes
> the `end_session` tool's `description` field — it's what teaches the model
> WHEN it's appropriate to hang up on itself. Consumed by
> `harp/interaction/session_tools.py`.

End the current conversation and put HARP back on standby. Call this when the visitor says goodbye, says they're done, or asks you to stop or close the session. Say a short spoken goodbye FIRST, then call this — it hangs up immediately. Do not call it while the visitor still needs help.
