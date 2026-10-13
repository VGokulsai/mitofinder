@echo off
rem Double-click: MitoFinder marks the mitochondria in every new picture in the "try" folder and opens the results.
rem Dragging one picture onto this file also works.
if "%~1"=="" (py -3 "%~dp0find_mitos.py" "%~dp0try") else (py -3 "%~dp0find_mitos.py" "%~1")
pause
