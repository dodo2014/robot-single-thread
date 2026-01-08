# -*- coding: utf-8 -*-
import logging
from logging.handlers import TimedRotatingFileHandler
from .path_helper import get_logs_dir
import datetime
import os


def setup_logger():
    """配置日志系统，按天切分日志文件"""
    # 确保logs目录存在
    # log_dir = "logs"
    log_dir = get_logs_dir()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件名格式：logs/robot_log_年_月_日.txt
    log_filename = os.path.join(log_dir, f"robot_log_{datetime.datetime.now().strftime('%Y_%m_%d')}.log")

    # 创建logger实例
    logger = logging.getLogger('RobotControl')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止重复输出

    # 清除已存在的处理器
    if logger.handlers:
        logger.handlers.clear()

    # 配置按天切分的文件处理器
    file_handler = TimedRotatingFileHandler(
        log_filename,
        when='midnight',  # 每天午夜切分
        interval=1,  # 切分间隔为1天
        backupCount=30,  # 保留30天的日志文件
        encoding='utf-8',
        utc=False  # 使用本地时间
    )

    # 设置日志格式
    log_formatter = logging.Formatter(
        '%(asctime)s - [%(threadName)s:%(thread)d] - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(log_formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)

    return logger


# 初始化全局logger
logger = setup_logger()