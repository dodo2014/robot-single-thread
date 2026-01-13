import json
import os
import sys
from src.utils.logger import logger
from .path_helper import get_config_path, get_base_path, get_vision_data_path

# 获取当前程序运行的根目录
# def get_app_path():
#     # path = os.path.dirname(sys.executable)
#     # logger.info(f"【打包模式】Exe路径: {sys.executable}")
#     # logger.info(f"【打包模式】根目录: {path}")
#
#     if "__compiled__" in globals() or getattr(sys, 'frozen', False):
#         # 如果是打包后的 exe 运行
#         # path = os.path.dirname(sys.executable)
#         # logger.info(f"【打包模式】Exe路径: {sys.executable}")
#         # logger.info(f"【打包模式】根目录: {path}")
#         #
#         # return os.path.dirname(sys.executable)
#         base_path = Path(sys.executable).parent
#
#     else:
#         # 如果是 python 脚本运行
#         # current_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/utils
#         # src_dir = os.path.dirname(current_dir)
#         # project_root = os.path.dirname(src_dir)
#         # return project_root # 或者项目根目录逻辑
#         base_path = Path(__file__).parent.parent.parent
#     return base_path

# base_path = get_base_path()
# CONFIG_FILE = base_path / "config.json"
# VISION_DATA_FILE = base_path / "vision_data.json"

CONFIG_FILE = get_config_path()
VISION_DATA_FILE = get_vision_data_path()

# 修改你的加载逻辑
# CONFIG_FILE = os.path.join(get_app_path(), "config.json")
# VISION_DATA_FILE = os.path.join(get_app_path(), "vision_data.json")

logger.info("config path: {}".format(CONFIG_FILE))
# logger.info("vision data path: {}".format(VISION_DATA_FILE))


class ConfigManager:
    def __init__(self, filepath="config.json"):
    # def __init__(self, filepath=CONFIG_FILE):
    #     self.filepath = filepath
        self.filepath = get_config_path(filepath)
        self.data = self.load()

    # 加载配置文件
    def load(self):
        if not os.path.exists(self.filepath):
            logger.error("配置文件不存在!")
            return {}
        try:
            logger.info(f"real filepath is {self.filepath}")
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}

    # 获取动作步骤配置
    # address: 动作的plc地址位
    def get_process_config(self, address):
        return self.data.get("processes", {}).get(str(address))

    # 获取动作的点位
    def get_process_points(self, address):
        return self.data.get("processes", {}).get(str(address)).get("points", [])
    # plc配置
    def get_plc_config(self):
        return self.data.get("plc_config", {})

    # 机器人参数配置
    def get_robot_params(self):
        return self.data.get("robot_params", {})

    # 原点配置
    def get_origin_params(self):
        return self.data.get("origin_params", {})

    # 产品配置
    def get_product_config(self):
        return self.data.get("product_config", {})

    # 当前产品型号
    def get_current_product_model(self):
        return self.data.get("product_config", {}).get("current_model", "Unknown")