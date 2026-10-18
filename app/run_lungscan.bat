@echo off
REM LungScan arayuzunu baslatir.
REM Model agirliklarini (best.pt, model_in_5_1.h5) bu klasore koyun.
cd /d "%~dp0"
python lungscan_gui.py
pause
