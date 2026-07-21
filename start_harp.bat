@echo off
cd /d "%~dp0"

echo Starting Harp App...
REM The main app now also does the gimbal head face tracking: it reads the
REM SHARED camera (no second RealSense open) and starts the face server on
REM 8788. Configure it in harp.yaml -> motion.gimbal_enabled / gimbal_port
REM (currently COM5). No separate "Gimbal" process anymore.
start "Harp App" powershell -NoExit -Command "uv run python -m harp"

echo Starting base-motor movement...
REM Base motors only (no --gimbal-port here — the head is the main app's job
REM now, and opening a camera here would fight the app for the RealSense).
start "Base Motors" powershell -NoExit -Command "uv run python -m harp.motion --left-port COM16 --right-port COM3"

echo All processes started!
REM The app opens the fullscreen face page itself once its face server is up
REM (harp.yaml -> motion.face_kiosk: true). Set that to false to open it here
REM instead with:  start msedge --kiosk http://127.0.0.1:8788/face.html --edge-kiosk-type=fullscreen
pause