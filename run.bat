@echo off
:: set code UTF-8
chcp 65001 > nul 2>&1
:: set code UTF-8

:: ==========================================
:: 1. 设置临时防误杀标题
:: 先给当前新打开的窗口一个临时标题，防止接下来的 taskkill 把自己也杀掉
:: ==========================================
title starting_robot_script_%random%

:: ==========================================
:: 2. 关闭旧的运行窗口及其子进程
:: 查找标题包含 "run robot-single-thread" 且进程名为 cmd.exe 的旧窗口
:: /F 强制关闭，/T 连带其子进程 (uv, python等) 一并关闭
:: ==========================================
echo 正在清理旧的 CMD 窗口...
taskkill /F /FI "IMAGENAME eq cmd.exe" /FI "WINDOWTITLE eq run robot-single-thread*" /T > nul 2>&1

:: ==========================================
:: 3. 兜底检查残留的 main.py 进程
:: (防止旧进程是在 VSCode 终端或其他非独立 cmd 窗口中运行的)
:: ==========================================
echo 正在检查是否残留 main.py 进程...
powershell -NoProfile -Command "$procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue; if (-not $procs) { $procs = Get-WmiObject Win32_Process -ErrorAction SilentlyContinue }; $procs | Where-Object { $_.CommandLine -match 'main\.py' -and ($_.Name -match 'python' -or $_.Name -match 'uv') } | ForEach-Object { Write-Host ('发现残留进程 ' + $_.Name + ' (PID: ' + $_.ProcessId + ')，正在兜底清理...'); Stop-Process -Id $_.ProcessId -Force }"
echo 进程清理完成。
echo ------------------------------

:: ==========================================
:: 4. 正式设置当前窗口的标题（为下一次重启做准备）
:: ==========================================
title run robot-single-thread

:: 5. 切换到目标目录
cd /d "D:\workspace\projects\robot-single-thread"

:: 检查目录切换是否成功
if %errorlevel% neq 0 (
    echo error：Can not switch to path D:\workspace\projects\robot-single-thread
    echo Please check the path exists or not！
    pause
    exit /b 1
)

:: 6. 执行uv run main.py命令
echo 正在执行命令：uv run main.py
echo ------------------------------
uv run main.py

:: 7. 命令执行完成后的处理
if %errorlevel% equ 0 (
    echo ------------------------------
    echo program complete！
) else (
    echo ------------------------------
    echo error：program run error！
)

:: 暂停窗口，方便查看执行结果（可根据需要删除）
pause