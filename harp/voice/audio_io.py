"""Microphone capture and speaker playback via sounddevice.

Deliberately vanilla: capture at the provider's required rate, play at the
rate the provider returns, and let the OS handle any device resampling. The
robust device-selection and VAD tricks from aura's SoundMonitor are a later
upgrade, not needed to prove the architecture.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

import sounddevice as sd

CHUNK_FRAMES = 1024  # frames per mic read (~64 ms at 16 kHz)


class Microphone:
    """Async context manager streaming raw 16-bit PCM mono chunks from the mic."""

    def __init__(self, rate: int, chunk_frames: int = CHUNK_FRAMES):
        self.rate = rate
        self.chunk_frames = chunk_frames
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream: sd.RawInputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> "Microphone":
        self._loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status):  # runs on a PortAudio thread
            if status:
                print(status, file=sys.stderr)
            # Hand bytes to the asyncio side thread-safely.
            self._loop.call_soon_threadsafe(self._queue.put_nowait, bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=self.rate, channels=1, dtype="int16",
            blocksize=self.chunk_frames, callback=callback,
        )
        self._stream.start()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    async def chunks(self):
        """Yield mic audio chunks forever (until the task is cancelled)."""
        while True:
            yield await self._queue.get()


class Speaker:
    """Async context manager that plays queued PCM audio, with clear-on-barge-in."""

    def __init__(self, rate: int):
        self.rate = rate
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream: sd.RawOutputStream | None = None
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> "Speaker":
        self._stream = sd.RawOutputStream(samplerate=self.rate, channels=1, dtype="int16")
        self._stream.start()
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc) -> None:
        # Cancel the writer AND wait for it to actually finish before touching
        # the stream. _run() offloads stream.write to a worker thread via
        # to_thread; cancel() only unblocks the awaiting task, it does not stop
        # a write already running on that thread. Closing the stream out from
        # under an in-flight write frees ALSA/PortAudio buffers mid-use — heap
        # corruption ("unaligned fastbin chunk", pa_linux_alsa.c failures).
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    async def _run(self) -> None:
        while True:
            data = await self._queue.get()
            # RawOutputStream.write blocks until there's buffer room; offload it.
            await asyncio.to_thread(self._stream.write, data)

    def play(self, pcm: bytes) -> None:
        self._queue.put_nowait(pcm)

    def clear(self) -> None:
        """Drop any queued-but-unplayed audio (used when the user interrupts)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
