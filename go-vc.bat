@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "HF_ENDPOINT=https://hf-mirror.com"

if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 venv，请先运行: D:\SoftWare\Python311\python.exe -m venv venv
    pause
    exit /b 1
)

venv\Scripts\python.exe -I vc.py %*
if errorlevel 1 (
    echo.
    echo [出错] 程序异常退出，请把上方错误信息截图反馈。
)
pause
