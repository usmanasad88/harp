# Wake-word Whisper initial_prompt

> Authoring notes stripped before use — see `harp/config.py`
> (`load_whisper_prompt`). This nudges the local faster-whisper model (used
> only to detect a wake word while HARP is idle, in `harp/listener/`) toward
> romanized Urdu script, so its output can match the romanized `wake_words`
> list in harp.yaml. Deliberately holds NO wake word — if Whisper ever echoes
> this prompt back on silence, that echo must not be able to cause a false
> wake. Override at runtime with the `HARP_WHISPER_PROMPT` env var without
> touching this file.

Yeh baat-cheet Roman Urdu aur English mein likhi gayi hai. Aap kaisay hain? Main theek hoon, shukriya.
