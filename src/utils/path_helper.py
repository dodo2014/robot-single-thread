
"""路径帮助工具，避免循环导入"""
import sys
import os
from pathlib import Path

def get_base_path() -> Path:
    """
    获取应用程序基础路径
    支持：开发模式、Nuitka打包、PyInstaller打包
    """
    # print(f"[Debug] sys.frozen: {getattr(sys, 'frozen', 'False')}")
    # print(f"[Debug] sys.executable: {sys.executable}")
    # print(f"[Debug] sys.argv[0]: {sys.argv[0]}")
    # print(f"[Debug] Env NUITKA_ONEFILE_BINARY: {os.environ.get('NUITKA_ONEFILE_BINARY')}")

    nuitka_binary = os.environ.get("NUITKA_ONEFILE_BINARY")
    if nuitka_binary:
        return Path(nuitka_binary).parent

    if getattr(sys, 'frozen', False):
        # # 打包后的可执行文件路径
        # if hasattr(sys, '_MEIPASS'):
        #     # PyInstaller打包
        #     base_path = Path(sys._MEIPASS)
        # else:
        #     # Nuitka打包
        #     base_path = Path(sys.executable).parent

        # 尝试获取原始 EXE 的路径
        # sys.argv[0] 在 Windows Nuitka Onefile 下通常是原始 exe 的绝对路径
        exe_path = Path(sys.argv[0]).absolute()

        # 为了保险，检查路径是否在临时文件夹中 (包含 "ONEFIL" 或 "Temp")
        # 如果 sys.argv[0] 被篡改指向了临时目录（极少见），我们需要回退方案
        if "ONEFIL" in str(exe_path) or "Temp" in str(exe_path):
            # 备用方案：Nuitka Onefile 启动时通常会将 CWD 设置为原始 EXE 目录
            return Path.cwd()

        return exe_path.parent
    # else:
    #     # 开发模式
    #     base_path = Path(__file__).parent.parent.parent

    # return base_path
    return Path(__file__).parent.parent.parent

def get_system_config_path() -> Path:
    """系统配置文件目录"""
    return get_base_path() / "config/system"

def get_vision_config_path() -> Path:
    """视觉配置文件目录"""
    return get_base_path() / "config/vision"

def get_config_path(config_file: str = "config.json") -> Path:
    """系统配置文件"""
    return get_base_path() / config_file
    # return get_system_config_path() / config_file

def get_vision_data_path(config_file: str = "vision_data.json") -> Path:
    """视觉坐标输出文件"""
    return get_base_path() / config_file
    # return get_system_config_path() / config_file

def get_logs_dir() -> Path:
    """日志目录路径"""
    logs_dir = get_base_path() / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir

def get_camera_img_dir() -> Path:
    img_dir = get_base_path() / "camera_img"
    img_dir.mkdir(exist_ok=True)
    return img_dir

def get_vision_detector_dir() -> Path:
    vision_detector_dir = get_base_path() / "src/depthSegmentPythonV3"
    return vision_detector_dir