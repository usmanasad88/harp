---
name: verify
description: How to drive HARP's changes end-to-end for verification — what can be driven without API keys/mic, and the working recipe for the web surfaces (dashboard + /user kiosk page).
---

# Verifying HARP changes

## What can be driven without secrets/hardware

- The **full agent** (`uv run python -m harp`) needs `.env` keys, a mic, and
  speakers — don't use it as the verification surface unless the change is in
  the voice path itself.
- The **dashboard web surfaces** (developer page `/`, end-user kiosk page
  `/user`) are fully drivable standalone: build the server on an ephemeral
  port around a fresh `Bus`, publish real events, and watch a real browser.
  This exercises the true surface (server → WS → JS state machine → pixels).

## Working recipe (verified 2026-07-17)

Script shape (run from a scratchpad file):

```python
sys.stdout.reconfigure(encoding="utf-8")   # console is cp1252; Urdu output crashes print otherwise
sys.path.insert(0, r"c:\Users\NCRA\harp")

from playwright.async_api import async_playwright
from harp.core.bus import Bus
from harp.dashboard.server import _build_server

bus = Bus()
async with _build_server(bus, "127.0.0.1", 8799,
                         get_app_state=lambda: "standby",
                         get_talk_key_held=lambda: False) as server:
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="msedge", headless=True)  # system Edge — no browser download
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto("http://127.0.0.1:8799/user")
        await asyncio.sleep(1.0)              # WS connect + connection seeds
        await bus.publish(SomeEvent(...))     # drive the page with real bus events
        await page.evaluate("document.body.className")  # /user: mode-idle|listening|thinking|responding|offline
        await page.screenshot(path="...")
```

Launch with `uv run --with playwright python script.py` — the `--with` overlay
keeps playwright OUT of the project venv (do not `uv add` it; see the
DEVLOG cv2 gotcha before touching deps at all).

## Gotchas

- The bus never replays: pages get current state only via the connection-seed
  getters (`get_app_state`, `get_talk_key_held`, `get_mic_muted`,
  `get_voice_tuning`) passed to `_build_server`/`serve`.
- `AgentSaid`/`UserSaid` finals can carry EMPTY text — the words are in the
  `final=False` deltas (see DEVLOG 2026-07-09). Publish deltas then a final.
- To probe kiosk reconnect: enter/exit `_build_server` manually
  (`__aenter__`/`__aexit__`) so the page outlives the server; reconnect
  backoff starts at 1 s.
