@echo off
cd /d E:\Git\valuation_grid
call .venv\Scripts\activate.bat
python market_summary.py
echo %date% %time% >> market_summary_log.txt
