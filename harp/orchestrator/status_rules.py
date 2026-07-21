"""THE RULE BOOK: which canned status line plays at which moment.

This is the one file to edit when you want HARP to say something different
(or nothing) at a life-cycle moment. The orchestrator, the error handler, and
the push-to-talk idle prompt all look their line up HERE by a stable "moment"
key — none of them hardcode a clip id.

HOW TO CHANGE WHAT HARP SAYS
----------------------------
  - Swap a line: change the clip id on the right-hand side (ids below).
  - Silence a moment: set its value to None (or delete the entry) — that
    moment then plays nothing, everything else keeps working.
  - Re-word a line: the WORDS live in scripts/generate_status_voice.py
    (LINES dict). Edit the text there, re-run that script (it re-renders the
    WAVs with Kokoro — instructions in its docstring), done. Ids stay stable,
    so this file doesn't change.
  - Add a brand-new line: add an id+text to scripts/generate_status_voice.py,
    re-run it, then reference the new id here.

AVAILABLE CLIP IDS (assets/status_voice/manifest.json — regenerate to change):
    starting_up             "Starting up."
    connection_established  "Connection established."
    ready                   "I'm ready."
    listening               "I'm listening."
    one_moment              "One moment, please."
    no_internet             "I can't reach the internet right now."
    connection_lost         "I lost my connection. Let me try again."
    retrying                "Something went wrong. Retrying."
    error_recoverable       "I ran into a problem. Give me a moment."
    error_fatal             "I couldn't recover from an error, and I have to shut down."
    mic_problem             "I'm having trouble with my microphone."
    hold_green_button       "Please hold the green button to talk to me."
    follow_no_person        "No known person detected in frame."
    follow_started          "I'm now following you. Please ensure a safe distance
                             from other people or obstacles."
    follow_stopped          "Follow mode stopped."
    going_standby           "Going on standby."
    session_ended           "Goodbye."
    shutting_down           "Shutting down. Goodbye."

MOMENT KEYS — when each fires (dotted = a variant; an unknown variant falls
back to its parent, e.g. a session_end with a cause nobody mapped plays
whatever plain "session_end" says):
"""

from __future__ import annotations

RULES: dict[str, str | None] = {
    # ---- boot -----------------------------------------------------------------
    # The very first thing a run says, before dialing the cloud.
    "boot": "starting_up",
    # Right after boot's connectivity probe: internet reachable / not.
    "boot.online": "connection_established",
    "boot.offline": "no_internet",
    #
    # ---- session end (ACTIVE → STANDBY) ---------------------------------------
    # One of these plays every time a conversation closes NORMALLY. The variant
    # is the machine-readable `cause` on EndOfInteractionDetected — error and
    # shutdown closes don't come through here (they play their own lines below).
    #
    # The person walked off (no face in frame for interaction.
    # absence_timeout_seconds). They're gone, so a goodbye is mostly for
    # bystanders — swap for "going_standby" if that feels odd.
    "session_end.walked_off": "session_ended",
    # Nobody said anything for interaction.silence_timeout_seconds. The person
    # may well still be standing there, so "Goodbye." tells them the
    # conversation is over (and, in push-to-talk mode, the idle prompt will
    # start inviting again shortly).
    "session_end.silence": "session_ended",
    # The model hung up on itself (the end_session tool) — it normally speaks
    # its own goodbye first, so a second "Goodbye." would be redundant.
    "session_end.agent": "going_standby",
    # The provider closed the stream (server-side session limit, clean network
    # drop). Nothing was said in-session, keep it neutral.
    "session_end.provider": "going_standby",
    # A human clicked "end session" on the dashboard. A deliberate, silent
    # close — keep it neutral (no goodbye, the operator knows what they did).
    "session_end.dashboard": "going_standby",
    # Fallback for any end whose cause isn't mapped above (or is empty —
    # e.g. an event published by an older/simpler caller).
    "session_end": "going_standby",
    #
    # ---- errors ---------------------------------------------------------------
    # Non-fatal errors: the orchestrator picks the variant from WHERE the error
    # came from (see _error_line in orchestrator.py), narrates, backs off,
    # and retries.
    "error.mic": "mic_problem",
    "error.connection": "connection_lost",
    "error.generic": "error_recoverable",
    # A fatal error: narrated once, then the app stops.
    "error.fatal": "error_fatal",
    #
    # ---- shutdown -------------------------------------------------------------
    # A requested, orderly shutdown (Ctrl+C, ShutdownRequested on the bus).
    "shutdown": "shutting_down",
    #
    # ---- push-to-talk idle prompt --------------------------------------------
    # Replayed every push_to_talk.idle_prompt_seconds while HARP is idle with
    # push-to-talk armed (harp/interaction/idle_prompt.py), so passers-by know
    # how to start a conversation. None here disables the prompt entirely.
    "idle_prompt": "hold_green_button",
    #
    # ---- follow mode (harp/motion/follow, the follow_person tool) -------------
    # A follow start was refused because face-ID sees no enrolled person.
    "follow.no_person": "follow_no_person",
    # Follow began: confirm + the safe-distance warning, in one clip.
    "follow.started": "follow_started",
    # Follow ended — asked to stop, lost sight of the person, or shutdown.
    "follow.stopped": "follow_stopped",
}


def line_for(moment: str) -> str | None:
    """Resolve a moment key to a clip id. Dotted keys fall back to their
    parent ("session_end.surprise" → "session_end") so an unmapped variant
    still says SOMETHING sensible; None means "stay silent for this moment"."""
    if moment in RULES:
        return RULES[moment]
    parent, _, _ = moment.rpartition(".")
    if parent and parent in RULES:
        return RULES[parent]
    return None
