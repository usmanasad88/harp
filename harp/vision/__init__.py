"""Vision: the camera and everything read from it.

Three concerns behind one shared camera:
  - camera.py    capture frames once, hand them to whoever needs them
  - face_id.py   recognize / log WHO is in frame → PersonIdentified
  - gestures.py  spot cues like a wave → GestureDetected

Frames also feed the live voice session (VoiceConnection.send_image) for image
Q&A. Face-ID and gestures publish to the bus; they don't call the voice or memory
layers directly. Vision on Gemini is stronger than on OpenAI — the provider
abstraction degrades gracefully, so this layer just supplies frames.
"""
