"""
HARP — Chunk 1 spike: minimal Gemini Live voice loop.

Purpose: talk to it and hear it reply, so we can judge spoken-Urdu quality
BEFORE building the full app. This is throwaway scaffolding, not the final
architecture — no RAG, no vision, no provider abstraction yet.

Run:
    pip install -r requirements.txt
    cp .env.example .env   # then put your GEMINI_API_KEY in it
    python spike_gemini_voice.py

Speak into your mic; it answers through your speakers. Ctrl+C to quit.
Try a few turns in Urdu and in English and see how the voice sounds.
"""

import asyncio
import os
import sys

import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# --- Knobs to experiment with -------------------------------------------------
# Gemini 3.1 Flash Live: native audio-to-audio, 90+ languages (Urdu covered),
# accepts image/video/text input, 128K context.
MODEL = "gemini-3.1-flash-live-preview"

VOICE = "Kore"          # other options include Puck, Charon, Aoede, Fenrir
LANGUAGE_CODE = "ur-IN"  # "ur-IN" for Urdu, "en-US" for English

SYSTEM_INSTRUCTION = (
    "You are HARP, a friendly humanoid assistant robot. "
    "Keep replies short, warm, and natural — you are speaking out loud. "
    "Reply in the same language the user speaks (English or Urdu)."
)

# Live API audio format is fixed: raw 16-bit PCM, mono.
INPUT_RATE = 16000   # what we send from the mic
OUTPUT_RATE = 24000  # what the model sends back
CHUNK = 1024         # frames per mic read
# -----------------------------------------------------------------------------


def build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    # v1beta is the API channel that exposes the Live / native-audio models.
    return genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})


CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=SYSTEM_INSTRUCTION,
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
        ),
        language_code=LANGUAGE_CODE,
    ),
)


class VoiceLoop:
    def __init__(self, session):
        self.session = session
        self.mic_queue: asyncio.Queue = asyncio.Queue()
        self.play_queue: asyncio.Queue = asyncio.Queue()

    async def capture_mic(self):
        """Read mic audio and queue it for sending."""
        loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            # Hand the raw bytes to the asyncio side thread-safely.
            loop.call_soon_threadsafe(self.mic_queue.put_nowait, bytes(indata))

        with sd.RawInputStream(
            samplerate=INPUT_RATE, channels=1, dtype="int16",
            blocksize=CHUNK, callback=callback,
        ):
            print("Listening… (speak; Ctrl+C to quit)")
            while True:
                await asyncio.sleep(0.1)

    async def send_mic(self):
        """Stream queued mic audio to the model."""
        while True:
            data = await self.mic_queue.get()
            await self.session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={INPUT_RATE}")
            )

    async def receive(self):
        """Receive model audio (and any text) and queue audio for playback."""
        while True:
            async for response in self.session.receive():
                if response.data:
                    self.play_queue.put_nowait(response.data)
                if response.text:
                    print(f"HARP: {response.text}", flush=True)

    async def play(self):
        """Play model audio through the speakers."""
        with sd.RawOutputStream(samplerate=OUTPUT_RATE, channels=1, dtype="int16") as out:
            while True:
                data = await self.play_queue.get()
                await asyncio.to_thread(out.write, data)


async def main():
    client = build_client()
    print(f"Connecting to {MODEL} (voice={VOICE}, language={LANGUAGE_CODE})…")
    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
        loop = VoiceLoop(session)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(loop.capture_mic())
            tg.create_task(loop.send_mic())
            tg.create_task(loop.receive())
            tg.create_task(loop.play())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye.")
