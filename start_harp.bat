@echo off
cd /d "%~dp0"

echo Starting Harp App...
start "Harp App" powershell -NoExit -Command "uv run python -m harp.app"

echo Starting Gimbal Motion...
start "Gimbal" powershell -NoExit -Command "uv run python -m harp.motion --gimbal-port COM5"

echo Starting Autonomous Patrol...
start "Autonomous Patrol" powershell -NoExit -Command "uv run python -m harp.motion.autonomous_patrol --left-port COM14 --right-port COM3 --side-length 3.0 --segments 2 --base-speed 500 --turn-speed 450"

echo All processes started!
REM 3. Wait a moment for the face server to boot, then open the face fullscreen
timeout /t 4 /nobreak >nul
start msedge --kiosk http://127.0.0.1:8788/face.html --edge-kiosk-type=fullscreen
pause