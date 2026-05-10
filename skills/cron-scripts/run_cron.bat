@echo off
cd /d E:\Git\valuation_grid
call .venv\Scripts\activate.bat
python skills\confidence-deviations\scripts\cron_job.py >> skills\cron-scripts\cron_log.txt 2>&1
echo %date% %time% == Task Completed == >> skills\cron-scripts\cron_log.txt
