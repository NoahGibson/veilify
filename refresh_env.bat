@echo off
rem ***********************************************************************
rem Removes existing virtual environment and creates new virtual
rem environment, installing the required packages
rem
rem This expects to be called in the following format:
rem   refresh_env
rem
rem ***********************************************************************

:: delete existing virtual environment (include, lib, scripts, tcl folders & pip-selfcheck.json)
echo Removing files...
RMDIR /S /Q Lib
RMDIR /S /Q Include
RMDIR /S /Q Scripts
RMDIR /S /Q tcl
del pip-selfcheck.json

:: create new virtual environment
echo Creating new environment...
python3 -m venv "%cd%"

:: install the required packages
echo Installing required packages...
call Scripts\activate.bat
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
call Scripts\deactivate.bat
