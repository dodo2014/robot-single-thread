import sys
import os
import ctypes
import platform
from PyQt5.QtWidgets import QApplication
from src.ui.hmi import MainHMI
from src.ui.config_window import ConfigEditorUI
from src.controller.controller_qt import Controller
from src.utils import logger
from src.utils import get_base_path


def disable_quickedit():
    """
    禁用 Windows CMD 快速编辑模式，防止鼠标点击导致程序假死
    """
    if platform.system() != 'Windows':
        return

    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32

        STD_INPUT_HANDLE = -10
        hStdIn = kernel32.GetStdHandle(STD_INPUT_HANDLE)

        mode = ctypes.c_uint()
        kernel32.GetConsoleMode(hStdIn, ctypes.byref(mode))

        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080

        new_mode = mode.value & ~ENABLE_QUICK_EDIT_MODE
        new_mode = new_mode | ENABLE_EXTENDED_FLAGS

        kernel32.SetConsoleMode(hStdIn, new_mode)
        print("已成功禁用 Windows 快速编辑模式。")
    except Exception as e:
        print(f"禁用快速编辑模式失败: {e}")


def main():
    disable_quickedit()
    logger.info(f"应用程序路径: {get_base_path()}")

    # 1. 启动 Qt 应用
    app = QApplication(sys.argv)

    # 2. 实例化控制器
    ctrl = Controller()

    # 3. 主界面: HMI 仪表盘 (工人操作界面)
    hmi = MainHMI(controller=ctrl)
    hmi.show()

    # window = ConfigEditorUI(controller_instance=ctrl)
    # window.show()

    # 4. 启动控制器线程 (后台业务逻辑)
    ctrl.start()

    # 5. 进入事件循环
    exit_code = app.exec_()

    # 6. 退出时优雅关闭
    ctrl.stop_service()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()