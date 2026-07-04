"""Memory: log conversations, summarize them per person, recognize returnees.

Everything said is logged (logger). When an interaction ends, the logs become a
short per-person memory summary (summarizer) saved in a store (store). When a
face is seen again, matcher finds that person's past memories so HARP can greet
and follow up in context. Follow-up intents captured here are what let a later
sighting re-open a session (see harp/triggers).

  - logger.py      append every turn of an interaction to a transcript
  - store.py       persist people ↔ embeddings ↔ summaries ↔ follow-ups
  - summarizer.py  on end-of-interaction, logs → a compact memory
  - matcher.py     match a detected face to a stored person
"""
