@echo off
rem PYNQ keyword demo. Double-click or run from a terminal; arguments go to
rem code\snn_keyword\run_board_demo.ps1 (e.g. -Single, -Kdot, -Wav file.wav).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0code\snn_keyword\run_board_demo.ps1" %*
pause
