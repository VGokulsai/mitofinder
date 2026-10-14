@echo off
rem Double-click: opens the placenta TEACHING pictures to click mitochondrion centres. Clicks save to data\placenta_crops\teach\answers.json.
py -3 "%~dp0click_centres.py" "%~dp0data\placenta_crops\teach"
pause
