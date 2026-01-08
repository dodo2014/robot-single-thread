@echo off
echo =================================
echo   SCARA Robot Build Script
echo =================================
echo.

echo Step 1: Cleaning previous builds...
if exist main.dist rmdir /s /q main.dist
if exist main.build rmdir /s /q main.build
if exist dist rmdir /s /q dist
echo .

echo Step 2: Building with Nuitka...
echo This may take a while...
echo.

:: 这里的 main.py 是你的入口文件
:: --standalone: 独立环境，生成一个包含所有依赖的文件夹
:: --onefile: 将文件夹压缩成一个单独的 .exe 文件
:: --mingw64: 指定编译器 (会自动下载)
:: --show-progress: 显示进度条
:: --show-memory: 显示内存使用
:: --enable-plugin=pyqt5: 启用 PyQt5 插件 (必须!)
:: --disable-console: 运行时不显示黑框 (CMD窗口)，调试时可以去掉这就话
:: --output-dir=dist: 输出目录
:: --windows-icon-from-ico=icon.ico: (可选) 设置图标，你需要有一个 icon.ico 文件

uv run python -m nuitka ^
    --standalone ^
    --mingw64 ^
    --show-progress ^
    --enable-plugin=pyqt5 ^
    --disable-console ^
    --output-dir=dist ^
    --no-pyi-file ^
    main.py

echo .
echo Step 3: Creating logs directory...

:: 定义生成的主目录变量
set TARGET_DIR=dist\main.dist

:: 创建日志目录
if not exist %TARGET_DIR%\logs mkdir %TARGET_DIR%\logs
:: 创建配置文件目录
if not exist %TARGET_DIR%\config mkdir %TARGET_DIR%\config
:: 创建系统配置目录
if not exist %TARGET_DIR%\config\system mkdir %TARGET_DIR%\config\system
:: 创建视觉配置目录
if not exist %TARGET_DIR%\config\vision mkdir %TARGET_DIR%\config\vision

echo .
echo Step 4: Copying config files..
:: 复制配置文件
copy config.json %TARGET_DIR%\ 2>nul
copy vision_data.json %TARGET_DIR%\ 2>nul
:: copy config\system\config.json %TARGET_DIR%\config\system\ 2>nul
:: copy config\system\vision_data.json %TARGET_DIR%\config\system\ 2>nul

copy README.md %TARGET_DIR%\ 2>nul

echo.
echo =================================
echo   Build complete!
echo   Please copy the ENTIRE folder:
echo   [ %TARGET_DIR% ]
echo   to the target machine.
echo =================================
echo.
dir /b dist\
pause

