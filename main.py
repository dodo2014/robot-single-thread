# import sys
# from src.utils.logger import logger
# from src.controller.controller import Controller
#
# def main():
#     logger.info("============== 系统启动 ==============")
#     try:
#         app = Controller()
#         app.start()
#     except Exception as e:
#         logger.critical(f"系统崩溃: {e}", exc_info=True)
#         sys.exit(1)
#
#
# if __name__ == "__main__":
#     main()


import sys
from PyQt5.QtWidgets import QApplication
from src.ui.config_window import ConfigEditorUI
from src.controller.controller_qt import Controller
from src.utils import logger
from src.utils import get_base_path


def main():
    logger.info(f"应用程序路径: {get_base_path()}")

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