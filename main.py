import sys
import os
import ctypes
import platform
from PyQt5.QtWidgets import QApplication
from src.ui.config_window import ConfigEditorUI
from src.controller.controller_qt import Controller
from src.utils import logger
from src.utils import get_base_path


# def disable_quickedit():
#     """禁用 Windows 控制台 QuickEdit 模式，防止点击 cmd 窗口导致程序卡死"""
#     try:
#         kernel32 = ctypes.windll.kernel32
#         kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 0x80)
#     except Exception:
#         pass


# def disable_quickedit():
#     import ctypes
#
#     kernel32 = ctypes.windll.kernel32
#
#     hstdin = kernel32.GetStdHandle(-10)
#
#     mode = ctypes.c_uint()
#
#     kernel32.GetConsoleMode(
#         hstdin,
#         ctypes.byref(mode)
#     )
#
#     ENABLE_QUICK_EDIT = 0x40
#     ENABLE_EXTENDED_FLAGS = 0x80
#
#     mode.value &= ~ENABLE_QUICK_EDIT
#     mode.value |= ENABLE_EXTENDED_FLAGS
#
#     kernel32.SetConsoleMode(
#         hstdin,
#         mode
#     )



def disable_quickedit():
    """
    禁用 Windows CMD 快速编辑模式，防止鼠标点击导致程序假死
    """
    if platform.system() != 'Windows':
        return

    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32

        # 获取标准输入句柄
        STD_INPUT_HANDLE = -10
        hStdIn = kernel32.GetStdHandle(STD_INPUT_HANDLE)

        # 获取当前控制台模式
        mode = ctypes.c_uint()
        kernel32.GetConsoleMode(hStdIn, ctypes.byref(mode))

        # 定义快速编辑模式的标志位
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080

        # 移除快速编辑标志
        new_mode = mode.value & ~ENABLE_QUICK_EDIT_MODE
        # 注意：在较新的 Windows 10/11 中，要修改 QuickEdit，通常也需要提供 ExtendedFlags
        new_mode = new_mode | ENABLE_EXTENDED_FLAGS

        # 设置新模式
        kernel32.SetConsoleMode(hStdIn, new_mode)
        print("已成功禁用 Windows 快速编辑模式。")
    except Exception as e:
        print(f"禁用快速编辑模式失败: {e}")

def main():
    disable_quickedit()
    logger.info(f"应用程序路径: {get_base_path()}")

    # app_path = get_base_path()
    # os.chdir(app_path)

    # 1. 启动 Qt 应用
    app = QApplication(sys.argv)

    # 2. 实例化 控制器 (此时还没启动线程)
    # Controller 内部会初始化 PLCClient
    ctrl = Controller()

    # 3. 实例化 界面
    # 将控制器传给界面，实现共享 PLC 和 数据交互
    window = ConfigEditorUI(controller_instance=ctrl)

    # 4. 启动 控制器线程 (后台跑业务逻辑)
    ctrl.start()

    # 5. 显示 界面 (前台跑配置和监控)
    window.show()

    # 6. 进入事件循环
    exit_code = app.exec_()

    # 7. 程序退出时，优雅关闭线程
    ctrl.stop_service()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()