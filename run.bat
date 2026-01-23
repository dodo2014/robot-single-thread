@echo off
:: set code UTF-8
chcp 65001 > nul 2>&1
:: set code UTF-8
title run robot-single-thread

:: 1. 切换到目标目录
cd /d "D:\workspace\projects\robot-single-thread"

:: 检查目录切换是否成功
if %errorlevel% neq 0 (
    echo error：Can not switch to path D:\workspace\projects\robot-single-thread
    echo Please check the path exists or not！
    pause
    exit /b 1
)

:: 2. 执行uv run main.py命令
echo 正在执行命令：uv run main.py
echo ------------------------------
uv run main.py

:: 3. 命令执行完成后的处理
if %errorlevel% equ 0 (
    echo ------------------------------
    echo program complete！
) else (
    echo ------------------------------
    echo error：program run error！
)

:: 暂停窗口，方便查看执行结果（可根据需要删除）
pause