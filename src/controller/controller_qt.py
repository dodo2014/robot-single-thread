from src.utils.logger import logger
from src.plc.plc_client import PLCClient
from src.utils.config_manager import ConfigManager, CONFIG_FILE, VISION_DATA_FILE
from src.utils.loading_index import LoadingIndex
from src.utils.kpi_counter import ProductionCounter
from src.utils.udp_client import UdpClient
from src.core.kinematics import ScaraKinematics
from src.core.trajectory import TrajectoryV2
from src.consts import const
from src.vision.detect_algo import DetectAlgoService
from src.core.tcp import compute_gripper_target
from PyQt5.QtCore import QThread, pyqtSignal
import time
import json
import os
import traceback
import copy
import socket
import math
import functools
import threading


# ===================== 定义带参数装饰器 =====================
def process_action(action_name: str, action_message: str = "", process_step: int = 0):
    """
    流程动作装饰器
    :param action_name: 流程名称，用于日志打印
    """

    def decorator(func):
        @functools.wraps(func)  # 保留原函数名称、文档等属性
        def wrapper(self, process_addr, value):
            # 1. 统一判断 value
            if value != 10:
                return
            # 2. 统一地址转换
            plc_addr = self.plc.map_modbus_address(process_addr)
            # 3. 统一日志打印
            logger.info(
                f"\n动作{hex(process_addr)} - {plc_addr} 收到请求 {value}，开始执行流程: {action_name}"
            )
            # 4. 判断当前步骤和当前动作
            if process_step in (const.process_step_remove_debris, const.process_step_loading,
                                const.process_step_blanking, const.process_step_detection,
                                const.process_step_palletizing):
                self.current_process_step = process_step
                self.current_process_msg = const.process_step_mapping[process_step]
            if action_message:
                self.current_action_msg = action_message

            logger.info(f"当前动作 {self.current_action_msg}, 当前步骤 {self.current_process_msg}")

            # 5. 执行原业务函数
            return func(self, process_addr, value)

        return wrapper

    return decorator


class Controller(QThread):
    log_signal = pyqtSignal(str)
    sig_wood_stick_alarm = pyqtSignal(int)  # 触发垫木报警弹窗 (传递层数)
    sig_wood_stick_clear = pyqtSignal()  # 关闭垫木报警弹窗
    sig_unloading_wood_stick_alarm = pyqtSignal()
    sig_unloading_wood_stick_clear = pyqtSignal()
    # HMI 相机画面信号 (color_img, result_img) — 从视觉线程发往主界面
    sig_camera_frame = pyqtSignal(object, object)
    # HMI 生产 KPI 信号 (today_count, cycle_time) — 每次放料成功 +1 时发射
    #   today_count: 今日累计产量; cycle_time: 当前节拍(秒), 无样本时为 0.0 (界面显示 "--")
    sig_kpi = pyqtSignal(int, float)

    def __init__(self):
        super().__init__()
        self.plc = None
        self.cfg = None

        self.cfg_manager = ConfigManager()
        self.robot_params = {}

        self.running = False
        self.is_estop_active = False  # 记录上一次急停状态
        # 内存缓存，格式: { "0x4008C": [[x,y,z,r], [x,y,z,r]...], ... }
        self.vision_data_cache = {}
        # 启动时尝试加载上次的数据
        self.load_vision_file()

        self.vision_service = None

        self.origin_point = self.cfg_manager.get_origin_params()  # 获取原点配置
        self.last_motion_end_point = self.origin_point  # 定义【全局当前坐标记录】，初始化为原点
        # 存储实时坐标[x,y,z,r]
        self.last_axis_status = [0.0, 0.0, 0.0, 0.0]
        # 存储实时电机状态 [J1, J2, J3, J4], 初始化为 0.0，供 UI 读取
        self.last_joint_status = [0.0, 0.0, 0.0, 0.0]

        self._motor_polling_running = False
        self._motor_polling_thread = None

        self.loop_count = 0

        # 上料拍照位置索引
        self.loading_index = LoadingIndex()

        # 是否需要进行顶层深度扫描(开机默认为True)
        self.need_rack_mapping = True

        self.current_action_msg = None
        self.current_process_step = 0
        self.current_process_msg = const.process_step_mapping[self.current_process_step]

        # 生产 KPI 统计 (今日产量 / 当前节拍), 数据持久化到 kpi.json
        self.kpi_counter = ProductionCounter()

    def load_config(self):
        try:
            with open(CONFIG_FILE, "r", encoding='utf-8') as f:
                cfg = json.load(f)
                self.cfg = cfg

            logger.info("配置加载成功")

        except Exception as e:
            logger.error(f"加载配置失败: {e}")

    def init_plc_client(self):
        cfg = self.cfg_manager.get_plc_config()
        self.plc = PLCClient(cfg["ip"], cfg["port"])
        logger.info(cfg)

    def _start_motor_poller(self):
        """启动电机状态独立轮询线程"""
        self._motor_polling_running = True
        self._motor_polling_thread = threading.Thread(target=self._motor_status_poller, daemon=True)
        self._motor_polling_thread.start()
        logger.info("电机状态轮询线程已启动 (200ms 间隔)")

    def _motor_status_poller(self):
        """独立线程：每 200ms 从 PLC 读取实时关节角度，更新 UI 显示变量"""
        while self._motor_polling_running:
            try:
                if self.plc and self.plc.is_connected:
                    regs = self.plc.read_holding_registers(
                        const.ADDR_FEEDBACK_START, const.FEEDBACK_LEN
                    )
                    if regs and len(regs) == const.FEEDBACK_LEN:
                        j1 = PLCClient.registers_to_float(regs[0:2])
                        j2 = PLCClient.registers_to_float(regs[2:4])
                        j3 = PLCClient.registers_to_float(regs[4:6])
                        j4 = PLCClient.registers_to_float(regs[6:8])

                        self.last_joint_status = [j1, j2, j3, j4]

                        # 轻量正运动学，更新笛卡尔坐标
                        fk_res = ScaraKinematics().forward_kinematics(
                            j1, j2, j3, j4,
                            self.l1, self.l2, self.z0, self.nn3
                        )
                        if fk_res:
                            self.last_axis_status = [
                                fk_res['x'], fk_res['y'],
                                fk_res['z'], fk_res['r']
                            ]
            except Exception:
                pass  # 轮询线程吞掉所有异常，绝不干扰主线程
            time.sleep(0.2)

    def run(self):
        # self.plc.connect()
        # 1. 在线程内部进行重量级初始化
        # 这样即使 PLC 连接超时，也不会卡住主界面
        self.robot_params = self.cfg_manager.get_robot_params()
        self.l1 = self.robot_params.get('l1')
        self.l2 = self.robot_params.get('l2')
        self.z0 = self.robot_params.get('z0')
        self.nn3 = self.robot_params.get('nn3')

        plc_cfg = self.cfg_manager.get_plc_config()
        self.plc = PLCClient(plc_cfg["ip"], plc_cfg["port"])

        logger.info("机器人后台控制服务启动...")

        logger.info("正在启动视觉服务...")
        try:
            # 获取当前产品型号
            current_product = self.cfg_manager.get_current_product_model()
            logger.info(f"current product: {current_product}")
            self.vision_service = DetectAlgoService(product_no=current_product)
            logger.info("视觉服务启动成功")
        except Exception as e:
            logger.error(f"视觉服务启动失败: {e}")
            # 即使失败，Controller 也要继续运行，不能退出

        if not self.plc.connect():
            logger.error("PLC 连接失败，线程退出")
            # return  # 连接失败直接退出线程

        self.reset_loding_index()
        self.running = True  # 标记运行

        self._start_motor_poller()  # 启动独立电机状态轮询线程

        while self.running:
            # if self.loop_count % 10 == 0:
            #     algo_response = self.vision_service.execute_detection_midian_depth(const.photo_type_loading)
            #     logger.info(f"algo_response >>>>>>>> : {algo_response}")
            #     # 将最新相机帧发送到 HMI 界面 (跨线程信号安全)
            #     last_color = getattr(self.vision_service, '_last_color_img', None)
            #     last_result = getattr(self.vision_service, '_last_result_img', None)
            #     logger.info(f"last result color >>>>>>>> : {last_result}")
            #     if last_color is not None or last_result is not None:
            #         logger.info(f">>>>>>>>>>>>> 发送最新相机帧 >>>>>>>>>>>>>")
            #
            #         self.sig_camera_frame.emit(last_color, last_result)

            try:
                self.loop_once()
                # 轮询间隔
                # time.sleep(2)
                self.loop_count += 1
                self.msleep(500)  # QThread 推荐用 msleep (毫秒)
            except KeyboardInterrupt:
                logger.info("用户停止程序")
                self.running = False
            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                # time.sleep(10)
                self.msleep(500)

    # 停止方法
    def stop_service(self):
        logger.info("正在停止控制服务...")
        self.running = False

        # ---- 新增：停止独立轮询线程 ----
        self._motor_polling_running = False
        if self._motor_polling_thread:
            self._motor_polling_thread.join(timeout=2.0)

        # 关闭视觉服务，释放相机资源
        if self.vision_service:
            self.vision_service.shutdown()

        self.wait()  # 等待线程安全退出
        logger.info("控制服务已停止")

    def reset_loding_index(self):
        self.loading_index.reset_search_index(const.search_index_name)
        self.loading_index.reset_search_index(const.head_layer_index_name)
        self.loading_index.reset_search_index(const.last_picked_layer_name, -1)
        self.loading_index.reload_search_index()

    # 配置热重载方法 (供界面调用)
    def reload_config(self):
        logger.info("收到配置更新信号，正在重载参数...")
        # 重新读取 ConfigManager
        self.cfg_manager = ConfigManager()  # 假设 ConfigManager 会重读文件
        self.robot_params = self.cfg_manager.get_robot_params()
        self.l1 = self.robot_params.get('l1')
        self.l2 = self.robot_params.get('l2')
        self.z0 = self.robot_params.get('z0')
        self.nn3 = self.robot_params.get('nn3')

        # 更新视觉服务的产品型号
        new_model = self.cfg_manager.get_current_product_model()
        if self.vision_service:
            self.vision_service.update_product(new_model)

    def check_estop(self):
        """
        检查是否触发急停
        :return: True(触发急停), False(正常)
        """
        try:
            # 读取急停状态
            addr = self.plc.map_modbus_address(const.ADDR_ESTOP_MONITOR)
            regs = self.plc.read_holding_registers(addr, 1)

            if regs and regs[0] == const.ESTOP_TRIGGER_VAL:  # 收到 10
                if not self.is_estop_active:
                    logger.critical("!!! 检测到急停信号 (0x400A8 - 168 = 10) !!! 系统进入急停状态 !!!")
                    self.is_estop_active = True
                    # 这里可以发送信号给 UI 弹窗提示
                return True
            else:
                if self.is_estop_active:
                    logger.info(">>> 急停解除，系统恢复运行 <<<")
                    self.is_estop_active = False
                return False
        except Exception as e:
            logger.error(f"急停检查异常: {e}")
            return False

    def check_and_handle_pause(self):
        """
        检查暂停状态。
        如果收到 10，则进入死循环阻塞，直到收到 11 或 触发急停。
        :return: True if paused and resumed, False if no pause happened
        """
        is_paused_once = False
        try:
            # 读取状态
            addr = self.plc.map_modbus_address(const.ADDR_PAUSE_CONTROL)
            regs = self.plc.read_holding_registers(addr, 1)

            if regs and regs[0] == const.VAL_PAUSE_REQ:  # 10
                logger.warning(">>> 监测到暂停信号(10)，系统挂起 <<<")

                while self.running:
                    # 优先检查急停 (急停权限高于暂停，必须能打断暂停)
                    if self.check_estop():
                        logger.critical("暂停期间触发急停，退出暂停等待")
                        return False  # 退出暂停函数，外层逻辑会捕获急停并 return

                    regs_wait = self.plc.read_holding_registers(addr, 1)
                    if regs_wait:
                        # logger.info(f"监测恢复信号：{regs_wait[0]}")
                        val = regs_wait[0]
                        # 判断恢复信号
                        if val == const.VAL_RESUME_REQ:  # 11
                            logger.info(">>> 监测到恢复信号(11)，复位信号并继续 <<<")
                            # 握手复位：写入 0
                            # self.plc.write_register(const.ADDR_PAUSE_CONTROL, const.VAL_RESET)
                            logger.info(
                                f"复位暂停信号 {hex(const.ADDR_PAUSE_CONTROL)} - {addr} -> 0")

                            is_paused_once = True
                            break  # 退出死循环

                    # 降低轮询频率
                    time.sleep(0.5)

        except Exception as e:
            logger.error(f"暂停检查出错: {e}")

        return is_paused_once

    def loop_once(self):
        # realtime_point = self.get_realtime_point(loop=1)

        # =======================================
        # 视觉测试
        # =======================================
        # try:
        #     alg_test_code = self.cfg_manager.get_alg_test_config()
        #     if alg_test_code != 0:
        # algo_response = self.vision_service.execute_detection_midian_depth(const.photo_type_find_head)
        # logger.info(f"algo_response >>>>>>>> : {algo_response}")
        # except Exception as e:
        #     logger.info(f"loop once algo_response >>>>>>>> : {e}\n{traceback.format_exc()}")

        # vision_result = self.vision_service.execute_detection(ptype=4)
        # logger.info(f"vision result: {vision_result}")

        # process_addr = 0x4008C
        # end_point = self.get_realtime_point()
        # loading = 1
        # photo_type = 4
        # handle_vision_res = self.handle_vision_recursive_v1(process_addr, end_point, loading, photo_type)
        # logger.info(f"handle_vision_res: {handle_vision_res}")
        # =======================================

        """执行一次完整的轮询和处理"""
        # 获取急停地址位的数据
        e_stop_regs = self.plc.read_holding_registers(self.plc.map_modbus_address(const.ADDR_ESTOP_MONITOR), 1)

        if self.loop_count % const.loop_log_rate == 0:
            # logger.info(f"current point: {realtime_point}")
            logger.info(
                f"emergency addr: {self.plc.map_modbus_address(const.ADDR_ESTOP_MONITOR)}, stop regs : {e_stop_regs}")

        # === 全局急停拦截 ===
        if self.check_estop():
            # 如果检测到急停：
            # 1. 清除可能存在的缓存状态
            self.last_motion_end_point = None
            # 2. 不读取业务寄存器，直接返回
            # 3. 打印日志提示（为了防止刷屏，可以加个状态位控制打印频率）
            if self.loop_count % const.loop_log_rate == 0: logger.info("系统急停中，等待复位...")
            time.sleep(0.5)  # 降低轮询频率
            return

        # 暂停检查
        self.check_and_handle_pause()

        # === 正常业务流程 ===
        # 1. 批量读取状态寄存器
        start_addr = self.plc.map_modbus_address(const.process_start_addr)
        states = self.plc.read_holding_registers(
            start_addr,
            const.process_num
        )

        if states is None or not states:
            return

        addr_value_map = {
            start_addr + idx: val
            for idx, val in enumerate(states)
        }
        # 防止日志刷屏
        if self.loop_count % const.loop_log_rate == 0:
            # logger.info(f"loop states: {states}")
            logger.info(f"loop states (地址:值): {addr_value_map}")

        # 2. 处理映射字典
        handler_map = {
            0x400A7: self.handle_process_0x400A7,
            0x40082: self.handle_process_0x40082,
            # 0x40083: self.handle_process_0x40083,
            0x40084: self.handle_process_0x40084,
            0x40085: self.handle_process_0x40085,
            0x40086: self.handle_process_0x40086,
            0x40087: self.handle_process_0x40087,
            0x40088: self.handle_process_0x40088,
            0x40089: self.handle_process_0x40089,
            0x4008A: self.handle_process_0x4008A,
            0x4008B: self.handle_process_0x4008B,
            0x4008C: self.handle_process_0x4008C,
            0x4008D: self.handle_process_0x4008D,
            0x4008E: self.handle_process_0x4008E,
            0x4008F: self.handle_process_0x4008F,
            0x40090: self.handle_process_0x40090,
            0x40091: self.handle_process_0x40091,
            0x40092: self.handle_process_0x40092,
            0x40093: self.handle_process_0x40093,
            # 0x40094: self.handle_process_0x40094,
            # 0x40095: self.handle_process_0x40095,
            # 0x40096: self.handle_process_0x40096,
            # 0x40097: self.handle_process_0x40097,
            # 0x40098: self.handle_process_0x40098,
            # 0x40099: self.handle_process_0x40099,
            # 0x4009A: self.handle_process_0x4009A,
            # 0x4009C: self.handle_process_0x4009C,
            # 0x400A8: self.handle_process_0x400A8,
        }

        # # 2. 遍历状态, 步长为 2; 址是 82, 84, 86，说明每个数据占 2 个字
        # for i in range(0, len(states)-1, 2):
        #     current_addr = const.process_start_addr + i
        #     # 获取两个寄存器的数据
        #     reg1 = states[i]
        #     reg2 = states[i + 1]
        #     if current_addr in handler_map:
        #         raw_values = (reg1, reg2)
        #
        #         # 调用对应的处理方法
        #         handler_map[current_addr](current_addr, raw_values)

        for i, val in enumerate(states):
            current_addr = const.process_start_addr + i

            if current_addr in handler_map:
                # 调用对应的处理方法
                handler_map[current_addr](current_addr, val)

        # 直接测试动作
        # self.handle_process_0x4008C(0x4008C, 10)

    def get_realtime_point(self, loop=0):
        """
        从 PLC 读取当前机械臂的实时关节角度，并通过正运动学转换为笛卡尔坐标。
        作为下一次插值轨迹的起点。
        """
        try:
            # 1. 批量读取 8 个寄存器 (4个轴 * 2寄存器/float)
            # 使用 map_modbus_address 确保地址映射正确
            regs = self.plc.read_holding_registers(const.ADDR_FEEDBACK_START, const.FEEDBACK_LEN)

            # 校验读取是否成功
            if not regs or len(regs) != const.FEEDBACK_LEN:
                logger.warning("读取实时反馈失败，回退使用内存记忆坐标")
                return self.last_motion_end_point

            # 2. 解析浮点数 (注意：汇川通常是 Little Word Endian)
            # 假设你的 self.plc.registers_to_float 已经封装好了 struct 处理
            # regs[0:2] -> J1, regs[2:4] -> J2 ...
            curr_j1 = self.plc.registers_to_float(regs[0:2])
            curr_j2 = self.plc.registers_to_float(regs[2:4])
            curr_j3 = self.plc.registers_to_float(regs[4:6])
            curr_j4 = self.plc.registers_to_float(regs[6:8])

            self.last_joint_status = [curr_j1, curr_j2, curr_j3, curr_j4]

            # 3. 调用正运动学 (FK) 计算 XYZR
            # 注意：需确保 ScaraKinematics 类中有 forward_kinematics 方法
            fk_res = ScaraKinematics().forward_kinematics(
                curr_j1, curr_j2, curr_j3, curr_j4,
                self.l1, self.l2, self.z0, self.nn3
            )

            if loop == 1:
                if self.loop_count % const.loop_log_rate == 0:
                    # logger.info("res: {}".format(regs))
                    logger.info(
                        f"PLC反馈关节信息: [{curr_j1:.2f}, {curr_j2:.2f}, {curr_j3:.2f}, {curr_j4:.2f}]")
                    logger.info(f"正运动计算结果: {fk_res}")
            else:
                # logger.info("res: {}".format(regs))
                logger.info(
                    # f"PLC反馈关节信息: J1={curr_j1:.2f}, J2={curr_j2:.2f}, J3={curr_j3:.2f}, J4={curr_j4:.2f}")
                    f"PLC反馈关节信息: [{curr_j1:.2f}, {curr_j2:.2f}, {curr_j3:.2f}, {curr_j4:.2f}]")
                logger.info(f"正运动计算结果: {fk_res}")

            if fk_res:
                # 4. 构造标准点位格式
                real_time_point = {
                    "name": "RealTime_Start",
                    "coords": [
                        fk_res['x'],
                        fk_res['y'],
                        fk_res['z'],
                        fk_res['r']
                    ],
                    "joints": [
                        curr_j1,
                        curr_j2,
                        curr_j3,
                        curr_j4
                    ],
                    "config": fk_res['config'],
                    "photo": 0  # 起点不拍照
                }
                # logger.info(f"获取实时起点成功: {real_time_point['coords']}")
                self.last_axis_status = [fk_res['x'], fk_res['y'], fk_res['z'], fk_res['r']]
                return real_time_point
            else:
                logger.error("正运动学计算失败，回退使用内存记忆坐标")
                return self.last_motion_end_point

        except Exception as e:
            logger.error(f"获取实时起点异常: {e}", exc_info=True)
            # 发生异常时，为了安全，回退到上一次记录的终点
            return self.last_motion_end_point

    def _fill_zero_group(self, a1, a2, a3, a4, a5, a6):
        """辅助函数：将一组地址清零"""
        self.plc.write_float(a1, 0.0)
        self.plc.write_float(a2, 0.0)
        self.plc.write_float(a3, 0.0)
        self.plc.write_float(a4, 0.0)
        self.plc.write_float(a5, 0.0)
        self.plc.write_float(a6, 0.0)

    # 批量发送动作下面的坐标数据，循环发送8个坐标
    # max_batch, 最多发送8个坐标
    def send_coords_batch(self, process_addr, points, max_batch=8):
        success_flag = True

        # 在发送前检查一下急停
        if self.check_estop():
            return False

        current_j2_angle = self.last_joint_status[1]

        # 维护一个"当前生效的姿态上下文"
        # 初始值：尝试从第一个点获取，如果第一个点是插值点(无config)，则沿用当前的
        active_config_type = None

        # 预扫描：看看这一批点里有没有明确指定 config 的关键点
        # 如果这一批点是 P_Start -> ... -> P_End，且 P_End 指定了 Up
        # 那么中间插值点最好也用 Up
        for pt in points:
            if pt.get("config"):
                active_config_type = pt.get("config")
                break  # 找到了就以它为准

        # 如果这一批里都没指定 (比如全是视觉点)，那就保持 None，后续走自动逻辑
        for i in range(max_batch):
            # 发送一个无意义的读取，仅仅为了保持 TCP 连接活跃
            self.plc.read_holding_registers(const.ADDR_FIRST, 1)

            addr_j1, addr_j2, addr_j3, addr_j4, addr_vel, addr_acc = const.point_addresses[i]

            # 每次循环都查一下急停，防止发送一半按急停
            if self.check_estop():
                logger.warning("发送过程中急停，终止发送")
                return False

            # 暂停检查
            self.check_and_handle_pause()

            if i < len(points):
                # === 有配置数据，进行计算 ===
                try:
                    pt_data = points[i]
                    # 获取笛卡尔坐标 [x, y, z, r]
                    coords = pt_data.get("coords", [0, 0, 0, 0])
                    xe, ye, ze, te = coords[0], coords[1], coords[2], coords[3]

                    # ik_res = ScaraKinematics.inverse_kinematics_v2(xe, ye, ze, te, self.l1, self.l2, self.z0, self.nn3)

                    # 检查配置里是否强制指定了 config (示教点优先使用配置)
                    # 1. 优先：点位自带配置 (示教点)
                    target_config = pt_data.get("config")

                    # 2. 次优：上下文配置 (插值点继承)
                    if not target_config:
                        target_config = active_config_type

                    # forced_config = pt_data.get("config")
                    ik_res = None

                    if target_config:
                        # 情况A：如果json中的point指定了elbow类型, 直接使用elbow数据
                        ik_res = ScaraKinematics.inverse_kinematics_v2(
                            xe, ye, ze, te, self.l1, self.l2, self.z0, self.nn3,
                            config_type=target_config
                        )
                        if ik_res:
                            active_config_type = target_config
                            current_j2_angle = ik_res['the2']
                    else:
                        # 情况B: 视觉点或插值点 (无 config)，使用智能选择
                        # 传入 current_j2_angle，让算法选一个最近的
                        ik_res = ScaraKinematics.calculate_best_inverse_kinematics(
                            xe, ye, ze, te, self.l1, self.l2, self.z0, self.nn3,
                            current_j2=current_j2_angle
                        )
                        # 选中后，更新 current_j2_angle，保证这一串点(插值序列)都不会突变
                        if ik_res:
                            current_j2_angle = ik_res['the2']
                            # 自动选出来的结果，也可以作为后续点的参考
                            active_config_type = ik_res['config']

                    if ik_res:
                        # 发送一个无意义的读取，仅仅为了保持 TCP 连接活跃
                        self.plc.read_holding_registers(const.ADDR_FIRST, 1)

                        # 写入浮点数 (关节角度/位置)
                        self.plc.write_float(addr_j1, ik_res['the1'])
                        self.plc.write_float(addr_j2, ik_res['the2'])
                        self.plc.write_float(addr_j3, ik_res['the3'])
                        self.plc.write_float(addr_j4, ik_res['th4'])
                        # 写入速度和加速度 (假设不需要特定控制，写0或默认值)
                        self.plc.write_float(addr_vel, 0.0)
                        self.plc.write_float(addr_acc, 0.0)

                        logger.info(
                            f"组{i + 1}: 坐标({xe},{ye},{ze}, {te}) -> "
                            f"关节({ik_res['the1']:.2f}, {ik_res['the2']:.2f}, {ik_res['the3']:.2f}, {ik_res['th4']:.2f})")
                    else:
                        logger.error(f"组{i + 1} 逆解失败: 目标点不可达 {coords}")
                        success_flag = False
                        # 逆解失败也填0，防止意外动作
                        self._fill_zero_group(addr_j1, addr_j2, addr_j3, addr_j4, addr_vel, addr_acc)
                except Exception as e:
                    logger.error(f" -> 插值点{i + 1} 计算异常: {e}")
                    success_flag = False
            else:
                # === 无配置数据 (超出 points 长度)，填充 0 ===
                # logger.debug(f"组{i+1}: 无数据，清零")
                self._fill_zero_group(addr_j1, addr_j2, addr_j3, addr_j4, addr_vel, addr_acc)

        return success_flag

    # 发送单个坐标，到地址[0x4006C, 0x4006E, 0x40070, 0x40072, 0x40074, 0x40076]
    def send_coords_once(self, process_addr, point):
        success_flag = True

        # 在发送前检查一下急停
        if self.check_estop():
            return False

        # 暂停检查
        self.check_and_handle_pause()

        point_address = const.point_once_address
        addr_j1, addr_j2, addr_j3, addr_j4, addr_vel, addr_acc = point_address
        name = point.get("name")
        coords = point.get("coords", [0, 0, 0, 0])
        elbow_config = point.get("config")
        xe, ye, ze, te = coords[0], coords[1], coords[2], coords[3]
        logger.info(f"\n发送坐标 {name}: {xe},{ye},{ze},{te}")
        try:
            # 逆解运算
            ik_res = ScaraKinematics.inverse_kinematics_v2(xe, ye, ze, te, self.l1, self.l2, self.z0, self.nn3,
                                                           config_type=elbow_config)
            if ik_res:
                # 发送一个无意义的读取，仅仅为了保持 TCP 连接活跃
                self.plc.read_holding_registers(const.ADDR_FIRST, 1)

                # 写入浮点数 (关节角度/位置)
                self.plc.write_float(addr_j1, ik_res['the1'])
                self.plc.write_float(addr_j2, ik_res['the2'])
                self.plc.write_float(addr_j3, ik_res['the3'])
                self.plc.write_float(addr_j4, ik_res['th4'])

                # 写入速度和加速度 (假设不需要特定控制，写0或默认值)
                self.plc.write_float(addr_vel, 0.0)
                self.plc.write_float(addr_acc, 0.0)

                logger.info(
                    f"坐标({xe},{ye},{ze},{te}) -> "
                    f"关节({ik_res['the1']:.2f}, {ik_res['the2']:.2f},{ik_res['the3']:.2f},{ik_res['th4']:.2f})")
                logger.info(f"发送完成\n")
            else:
                logger.error(f"逆解失败：动作{process_addr}目标点{coords}不可达")
                success_flag = False
                # 逆解失败也填0，防止意外动作
                self._fill_zero_group(addr_j1, addr_j2, addr_j3, addr_j4, addr_vel, addr_acc)
        except Exception as e:
            logger.error(f"发送plc异常: 动作{process_addr}目标点{coords}, 异常：{e}")
            success_flag = False

        return success_flag

    # 监听坐标的plc到位返回值，12表示到位
    def monitor_plc_ok(self, process_addr, point):
        start_time = time.time()
        while time.time() - start_time < 10.0 and self.running:
            res = self.plc.read_holding_registers(process_addr, 1, unit=1)
            if not res.isError() and res.registers and res.registers[0] == 12:
                logger.info(f"动作{process_addr},坐标{point} 执行到位，plc写入12")
                return
            time.sleep(0.1)

    def prepare_params_for_camera(self, point_config):
        """
        将配置点转换为相机需要的格式 (X, Y, Z, World_R)
        :param point_config: 包含 coords=[x,y,z,r] 和 config='elbow_up' 的字典
        """
        coords = point_config.get("coords", [0, 0, 0, 0])
        xe, ye, ze, te = coords
        cfg_type = point_config.get("config", "elbow_up")  # 默认为 up

        # 1. 逆解计算 J1, J2
        # 注意：te 此时是相对角度 (电机角度)
        ik_res = ScaraKinematics.inverse_kinematics_v2(
            xe, ye, ze, te,
            self.l1, self.l2, self.z0, self.nn3,
            config_type=cfg_type
        )

        if not ik_res:
            logger.error("相机参数准备失败：逆解失败")
            return None

        j1 = ik_res['the1']
        j2 = ik_res['the2']
        j4_relative = ik_res['th4']  # 即传入的 te

        # 2. 计算世界绝对角度
        # World_R = J1 + J2 + J4_relative
        world_r = j1 + j2 + j4_relative

        # 归一化 (可选)
        while world_r > 180: world_r -= 360
        while world_r <= -180: world_r += 360

        return [xe, ye, ze, world_r]

    def process_camera_result_to_plc_data(self, camera_result_coords):
        """
        将相机返回的绝对坐标数据，转换为 PLC 可用的相对角度数据
        :param camera_result_coords: [x, y, z, world_r]
        :return: 包含相对角度的目标点字典
        """
        target_x, target_y, target_z, target_world_r = camera_result_coords

        # 1. 获取当前机械臂状态 (用于智能决策姿态)
        # 假设 self.last_joint_status = [j1, j2, j3, j4]
        current_j2 = self.last_joint_status[1]

        # 2. 智能逆解 (自动决定是 Up 还是 Down)
        # 注意：这里传入的 te 参数暂时不重要，因为我们只需要 J1 和 J2
        # 我们随便传个 0，反正后面会重新算 J4
        best_ik = ScaraKinematics.calculate_best_inverse_kinematics(
            target_x, target_y, target_z, 0,  # te 传 0
            self.l1, self.l2, self.z0, self.nn3,
            current_j2=current_j2  # 关键：传入当前角度做参考
        )

        if not best_ik:
            logger.error(f"视觉点不可达: {camera_result_coords}")
            return None

        # 3. 获取最优解的 J1, J2
        j1_new = best_ik['the1']
        j2_new = best_ik['the2']

        # 4. 反算 J4 相对角度 (电机角度)
        # 调用之前写的辅助函数: J4 = World_R - (J1 + J2)
        j4_relative_new = self.calculate_j4_from_world_angle(
            j1_new, j2_new, target_world_r
        )

        # 5. 组装结果
        # 注意：这里我们算出了 config，最好把它记下来，传给 send_coords_batch
        # 这样插值的时候也会遵循这个 config
        final_point = {
            "name": "Vision_Target",
            "coords": [target_x, target_y, target_z, j4_relative_new],
            "config": best_ik['config'],  # 'elbow_up' 或 'elbow_down'
            "photo": 0
        }

        logger.info(f"视觉解算结果: Config={best_ik['config']}, J4电机角度={j4_relative_new:.2f}")
        return final_point

    def calculate_j4_from_world_angle(self, j1, j2, target_world_r):
        """
        根据给定的 J1, J2 和目标世界角度，反算 J4 电机角度
        公式: J4 = World_R - (J1 + J2)
        """
        # 1. 基础反算
        j4 = target_world_r - (j1 + j2)

        # 2. 归一化处理 (限制在 -180 到 180 之间)
        # 这一步非常重要，确保电机走最短路径，且数值符合常规逻辑
        while j4 > 180:
            j4 -= 360
        while j4 <= -180:
            j4 += 360

        return j4

    def move_forward(self, point, distance=0):
        """
        末端坐标前移，包括前进，后退
        :param point: 坐标点
        :param distance: 移动距离，>0表示前进，<0表示后退
        :return: 移动后的点坐标
        """
        try:
            xe, ye, ze, te = point.get("coords")
            config_curr = point.get("config")
            # ik_res = ScaraKinematics().inverse_kinematics_v2(xe, ye, ze, te, self.l1, self.l2, self.z0, self.nn3,
            #                                                  config_type=config_curr)
            # j1_curr = ik_res["the1"]
            # j2_curr = ik_res["the2"]
            #
            # target_x, target_y, target_z, target_r = ScaraKinematics().calculate_forward_move(self.l1, self.l2, self.z0,
            #                                                                                   self.nn3, xe, ye, ze, te,
            #                                                                                   j1_curr, j2_curr,
            #                                                                                   distance,
            #                                                                                   config_curr=config_curr)
            target_x, target_y, target_z, target_r = ScaraKinematics().calculate_forward_move(self.l1, self.l2, self.z0,
                                                                                              self.nn3, xe, ye, ze, te,
                                                                                              distance,
                                                                                              config_curr=config_curr)
            name = "FP_P0"  # forward point
            if distance < 0:
                name = "BP_P0"  # backward point

            foward_point = {
                "name": name,
                "coords": [
                    target_x,
                    target_y,
                    target_z,
                    target_r
                ],
                "photo": 0,
                "config": config_curr
            }
            logger.info(f"foward point: {foward_point}")
            return foward_point
        except Exception as e:
            logger.error(f"move forward error: {e}")

    def move_up_down(self, point, distance=0):
        try:
            xe, ye, ze, te = point.get("coords")
            config_curr = point.get("config")

            target_x = xe
            target_y = ye
            target_z = ze + distance
            target_r = te

            name = "UP_P0"
            if distance < 0:
                name = "DOWN_P0"

            new_point = {
                "name": name,
                "coords": [
                    target_x,
                    target_y,
                    target_z,
                    target_r
                ],
                "photo": 0,
                "config": config_curr
            }
            return new_point
        except Exception as e:
            logger.error(f"move up down error: {e}")

    def take_photo(self):
        # 执行拍照动作，拍照结果成功返回OK，异常返回NG, 返回字符串
        return "OK"

    def take_photo_check(self):
        return "OK"

    def decode_algo_result(self, result: dict, ptype: int):
        exists = result.get("exists")
        res = "error"

        if ptype == const.photo_type_normal:  # 1/有料，报ok
            if exists == 1:
                res = "ok"
            else:
                res = "empty"
        elif ptype == const.photo_type_loading:  # 1/有料，报ok; 2/无料, 报ng
            if exists == 1:
                res = "ok"
            else:
                res = "empty"
        elif ptype == const.photo_type_find_head:
            if exists == 1:
                res = "ok"
            else:
                res = "empty"
        elif ptype == const.photo_type_unloading:  # 1
            if result.get("coords", []):
                res = "ok"
            else:
                res = "empty"
        elif ptype == const.photo_type_aluminum:  # 1/有铝屑，报ng; 2/没铝屑，正常，报ok
            if exists == 1:  # 有铝屑，真报错
                res = "error"
            elif exists == 2:  # 没有铝屑，返回OK
                res = "ok"
        elif ptype == const.photo_type_cylinder:  #
            state = result.get("state")
            if exists == 1 and state == 2:  # 1/松开状态，2/夹紧状态
                res = "ok"
            else:
                res = "error"

        return res

    def take_photo_position(self, point_coords, config, loading=None, ptype=const.photo_type_normal):
        """
        :param point_coords:传给相机的拍照位置坐标
        :param config, 肘关节的状态，elbow_up/elbow_down
        :param loading, 上下料参数，1/上料，2/下料
        :param ptype, 拍照触发的动作类型，普通拍照(物料识别)/1，上料(空料判断)/2，下料(满料判断)/3，铝屑识别/4
        :return:
        {
            "res": "ok/error/empty"
            "coords" : [
             [x,y,z,r],
             [x,y,z,r],
            ],
            "trigger":"retrive/photo" # retrive,抓取，两个坐标; photo拍照，一个坐标
        }

        photo，表示coords中返回一个坐标p_r0，直接移动到p_r0处进行拍照
        retrive，表示coords中返回两个坐标p_r0，p_r1，移动到p_r1处进行抓取

        """
        try:
            if not self.vision_service:
                logger.error("视觉服务未就绪")
                return {"res": "error", "coords": [], "trigger": ""}

            logger.info(f"photo position coords >>>>>>>> : {point_coords}, loading : {loading}, ptype : {ptype}")

            # camera_prepare_coords = self.prepare_params_for_camera({"coords": camera_coords, "config": config})
            # logger.info(f"camera_prepare_coords >>>>>> : {camera_prepare_coords}")
            # pos = VisionSystem().run(camera_prepare_coords, loading=loading) # 相机只要x,y,z，不要r参数

            # algo_response = self.vision_service.execute_detection(ptype)
            algo_response = self.vision_service.execute_detection_midian_depth(ptype, check_estop_func=self.check_estop)
            logger.info(f"algo_response >>>>>>>> : {algo_response}")

            # 将最新相机帧发送到 HMI 界面 (跨线程信号安全)
            last_color = getattr(self.vision_service, 'last_color_img', None)
            last_result = getattr(self.vision_service, 'last_result_img', None)
            logger.info(f"last result color >>>>>>>> : {last_result}")
            if last_color is not None or last_result is not None:
                logger.info(f">>>>>>>>>>>>> 发送最新相机帧 >>>>>>>>>>>>>")

                self.sig_camera_frame.emit(last_color, last_result)

            if algo_response["code"] == 0:
                result = algo_response["result"]
                # { "ok": 1, "coords": [x, y, z, r], "type": "retrieve" }
                # 或者如果是多目标: "coords": [[x,y,z,r], [x,y,z,r]]
                # 兼容之前的 handle_vision_recursive 逻辑，我们需要适配格式
                detected_coords = result.get("coords", [])
                # 如果返回的是单层列表 [x,y,z,r]，转为嵌套 [[x,y,z,r]]
                if detected_coords and isinstance(detected_coords[0], (int, float)):
                    detected_coords = [detected_coords]

                trigger_type = "retrieve"  # 默认抓取

                layer = result.get("layer")
                position = result.get("position")
                res = self.decode_algo_result(result, ptype)

                return {
                    "res": res,
                    "coords": detected_coords,
                    "trigger": trigger_type,
                    "layer": layer,
                    "position": position
                }
            else:
                logger.error(f"视觉识别失败: {algo_response.get('err_msg')}")
                return {"res": "error", "coords": [], "trigger": ""}

        except Exception as e:
            logger.info(f"take photo position error: {e}, traceback: {traceback.format_exc()}")

        return {"res": "error", "coords": [], "trigger": ""}

    def save_vision_data(self, process_addr, coords_list, photo_type=None, layer=None, index=None):
        """
        保存视觉坐标数据
        :param process_addr, 动作地址 (int)，如 0x4008C
        :param coords_list, 坐标列表 [[x,y,z,r], ...]
        :param photo_type, 拍照动作类型
        :param layer, 物料层号， 下料用
        :param index, 当前层的物料索引号， 下料用
        """
        key = hex(process_addr)  # 转成字符串 "0x4008c" 作为 Key

        # 校验photo_type
        if photo_type is None or photo_type not in const.PHOTO_TYPE_DESC:
            logger.warning(f"保存失败：无效的photo_type={photo_type}")
            return

        # 1. 更新内存
        if key not in self.vision_data_cache:
            self.vision_data_cache[key] = {}

        # self.vision_data_cache[key][photo_type] = coords_list
        self.vision_data_cache[key][photo_type] = {
            "coords": coords_list,
            "layer": layer,
            "position": index
        }

        # 2. 更新文件 (全量保存，防止覆盖其他动作的数据)
        try:
            # 先读取现有文件内容（如果有）
            current_data = {}
            if os.path.exists(VISION_DATA_FILE):
                with open(VISION_DATA_FILE, 'r', encoding='utf-8') as f:
                    try:
                        current_data = json.load(f)
                    except json.JSONDecodeError:
                        pass

            # 确保顶层key存在
            if key not in current_data:
                current_data[key] = {}

            # 更新当前动作的数据
            current_data[key][str(photo_type)] = {
                "desc": const.PHOTO_TYPE_DESC[photo_type],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "coords": coords_list,
                "layer": layer,
                "position": index
            }

            # 写入文件
            with open(VISION_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, indent=4, ensure_ascii=False)

            logger.info(
                f"视觉数据已保存: 地址={key}, 类型={photo_type}({const.PHOTO_TYPE_DESC[photo_type]}), 坐标数={len(coords_list)}")

        except Exception as e:
            logger.error(f"保存视觉数据失败: {e}")

    def get_vision_data(self, process_addr, photo_type=None):
        """
        获取视觉坐标数据
        :param process_addr: 产生数据的动作地址 (int)
        :return: coords_list 或 []
        """
        key = hex(process_addr)

        # 必须传入photo_type
        if photo_type is None:
            logger.warning("获取数据失败：未指定photo_type")
            return []

        # 1. 优先从内存读
        if key in self.vision_data_cache and photo_type in self.vision_data_cache[key]:
            data = self.vision_data_cache[key][photo_type]
            return data.get("coords", [])

        # 2. 内存没有，尝试从文件重新加载 (应对程序重启的情况)
        self.load_vision_file()

        # 再次从缓存读取
        if key in self.vision_data_cache and photo_type in self.vision_data_cache[key]:
            data = self.vision_data_cache[key][photo_type]
            return data.get("coords", [])

        logger.warning(f"未找到视觉数据: 地址={key}, 类型={photo_type}")
        return []

    def get_vision_data_full(self, process_addr, photo_type=None):
        """
       【新增方法】获取完整数据（包含 coords、layer、index）
        :return: dict {"coords": [...], "layer": x, "position": x}
        """
        key = hex(process_addr)

        if photo_type is None:
            logger.warning("获取完整数据失败：未指定photo_type")
            return {"coords": [], "layer": None, "position": None}

        # 内存读取
        if key in self.vision_data_cache and photo_type in self.vision_data_cache[key]:
            return self.vision_data_cache[key][photo_type]

        # 从文件加载
        self.load_vision_file()

        if key in self.vision_data_cache and photo_type in self.vision_data_cache[key]:
            return self.vision_data_cache[key][photo_type]

        logger.warning(f"未找到完整视觉数据: 地址={key}, 类型={photo_type}")
        return {"coords": [], "layer": None, "position": None}

    def load_vision_file(self):
        """从文件加载视觉坐标数据到内存"""
        if not os.path.exists(VISION_DATA_FILE):
            return
        try:
            with open(VISION_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 清空原有缓存，重新加载
            self.vision_data_cache.clear()
            # 解析嵌套结构：地址 -> photo_type -> 坐标
            for addr_key, type_dict in data.items():
                # 初始化地址节点
                self.vision_data_cache[addr_key] = {}
                # 遍历photo_type
                for photo_type_str, value in type_dict.items():
                    photo_type = int(photo_type_str)
                    # self.vision_data_cache[addr_key][photo_type] = value.get("coords", [])
                    self.vision_data_cache[addr_key][photo_type] = {
                        "coords": value.get("coords", []),
                        "layer": value.get("layer"),
                        "position": value.get("position")
                    }

        except Exception as e:
            logger.error(f"读取视觉文件失败: {e}")

    def get_vision_status(self):
        """
        获取状态，判断物料加工状态，以及垫木取/放状态
        11：物料未加工完成,还有剩余物料  12：加工结束，所有的物料都加工结束；13：取垫木；14：取垫木结束
        """
        return 11

    # 辅助：阻塞监听 PLC 寄存器直到变为指定值 (支持超时，但等待复位时通常超时时间设很长或无限)
    def wait_for_plc_val(self, addr, target_val, timeout=7200.0):
        start_time = time.time()
        logger.info(f"阻塞监听PLC，开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        count = 0
        while self.running:
            # 发送一个无意义的读取，仅仅为了保持 TCP 连接活跃
            self.plc.read_holding_registers(const.ADDR_FIRST, 1)

            # 1. 优先检查急停
            if self.check_estop():
                logger.warning("任务因急停中断，停止等待")
                return False  # 返回 False，上层逻辑会感知并退出

            # 暂停检查, 直到恢复才继续运行；暂停恢复后，重置 start_time，给足时间让机械臂继续走
            if self.check_and_handle_pause():
                start_time = time.time()
                logger.info(
                    f"阻塞监听PLC，暂停恢复，重置开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

            time_consume = time.time() - start_time
            if count % 20 == 0: logger.info(f"阻塞监听PLC，耗时 {time_consume} s")
            # 如果 timeout < 0，则无限等待
            if timeout > 0 and (time_consume > timeout):
                logger.error(f"等待PLC地址 {hex(addr)} 变为 {target_val} 超时")
                return False

            # 读取当前值
            regs = self.plc.read_holding_registers(addr, 1)
            if regs and regs[0] == target_val:
                return True

            count += 1
            time.sleep(0.1)  # 避免死循环占用过高CPU
        return False

    def _move_segment_to_target(self, process_addr=None, start_point=None, target_point=None, interpolate=False):
        """
        辅助函数：从当前位置移动到目标点 (包含获取起点、插值、发送、握手)
        :param process_addr: PLC地址位
        :param start_point: 起始点
        :param target_point: 目标点
        :param interpolate: 是否插值
        """
        # 1. 获取实时起点
        if not start_point:
            start_point = self.get_realtime_point()

        # 判断是否需要插值：
        # 需要插值，先插值，再发送坐标
        # 不需要插值，直接发送坐标
        target_name = target_point.get("name", "Temp_Point")

        if interpolate:
            # 2. 生成插值路径
            interpolated_path = TrajectoryV2().generate_cartesian_interpolated_path(
                [start_point, target_point],
                num_inserts=const.point_interpolated_num
            )
            points_to_send = interpolated_path[1:]  # 去掉起点

            logger.info(f"执行移动片段 -> {target_name}")

            # 3. 发送数据
            if not self.send_coords_batch(process_addr, points_to_send, 8):
                # 1. 检查急停
                if self.check_estop():
                    return False  # 直接退出函数，即清除了当前流程数据

                logger.error("坐标发送失败")
                return False
        else:
            # 3. 发送数据
            if not self.send_coords_once(process_addr, target_point):
                # 1. 检查急停
                if self.check_estop():
                    return False  # 直接退出函数，即清除了当前流程数据

                logger.error("坐标发送失败")
                return False

        if self.check_estop(): return False  # 确认是否因急停退出
        # 4. 握手 (11)
        self.plc.write_register(process_addr, 11)
        logger.info(f"plc握手，写入11")

        # 5. 等待到位 (12), timeout=-1表示无限等待
        if not self.wait_for_plc_val(process_addr, 12, timeout=7200):
            if self.check_estop(): return False  # 再次确认是否因急停退出

            logger.error(f"执行移动片段 -> {target_name} 等待到位超时")
            return False
        logger.info(f"收到plc反馈12")

        # 6. 更新内存中的位置记录
        self.last_motion_end_point = target_point
        return True

    def transform_tool_coord(self, coord, align_camera=0, align_z_diff=0, joint_valid=True):
        """
        工具坐标转换成法兰坐标
        工具默认使用夹爪对齐;
        align_camera=1的情况下，使用相机对齐

        :param coord: 相机坐标系的坐标
        :param align_camera:
            欺骗参数，align_camera=1， 让【相机】成为末端工具去对齐物料， 把 gripper_offset 设置为 camera_offset，
            同时使用 align_z_diff 代替z_diff
        """
        try:
            robot_params = self.robot_params

            # 1. 获取配置
            tool_cfg = self.cfg_manager.get_current_tool_model()
            cam_cfg = tool_cfg.get("camera", {})
            grip_cfg = tool_cfg.get("main_gripper", {})  # 获取这组夹爪的配置

            # 2. 准备 Offset 参数
            camera_offset = [cam_cfg.get("offset_x", 0), cam_cfg.get("offset_y", 0)]

            # 【关键】这里的 Offset 是 两个夹爪的连线中点 相对于 电机中心 的偏移
            gripper_offset = [grip_cfg.get("offset_x", 0), grip_cfg.get("offset_y", 0)]
            z_diff = grip_cfg.get("z_diff", 0)
            gripper_install_angle = grip_cfg.get("gripper_install_angle", 0)
            angle_offset = grip_cfg.get("angle_offset", 0)

            # 获取当前位置数据
            real_time_point = self.get_realtime_point()
            logger.info(f"real_time_point:{real_time_point}")

            vision_data = coord
            robot_state = self.last_axis_status
            robot_joints = self.last_joint_status
            elbow_config = real_time_point.get("config", "elbow_up")

            if align_camera:
                logger.info("cheat camera...")
                gripper_offset = camera_offset
                z_diff = align_z_diff

            # 4. 调用计算
            target_coord = compute_gripper_target(
                camera_data=vision_data,
                robot_state=robot_state,
                elbow_config=elbow_config,
                robot_joints=robot_joints,
                camera_offset=camera_offset,
                gripper_offset=gripper_offset,  # 传入中点偏移
                z_diff=z_diff,
                robot_params=robot_params,
                cam_rotation=cam_cfg.get("rotation", 0),
                gripper_install_angle=gripper_install_angle,
                angle_offset=angle_offset,
                joint_valid=joint_valid,
            )

            return target_coord
        except Exception as e:
            logger.error(f"工具坐标转换错误 {e}\n{traceback.format_exc()}")

        return []

    def handle_vision_recursive_v0(self, process_addr, point, loading=None, photo_type=const.photo_type_normal):
        """
        :param process_addr: 动作地址位
        :param point: 拍照坐标
        :param loading, 上下料参数，1/上料，2/下料
        :param photo_type, 拍照触发的动作类型，普通拍照(物料识别)/1，上料(空料判断)/2，下料(满料判断)/3，铝屑识别/4, 默认1

        功能:
            递归视觉逻辑
            内部会处理：拍照 -> (可能移动 -> 重拍) -> 移动到抓取点
        1. Photo 模式：移动到新拍照点，继续循环。
        2. Retrieve 模式：保存坐标数据，返回 True，不执行抓取移动。
        """
        loop_count = 0
        max_loops = 5

        # 初始化 camera_coords
        camera_coords = point.get("coords", [])
        config = point.get("config", "elbow_up")
        loading = loading

        while self.running and loop_count < max_loops:
            loop_count += 1
            logger.info(
                f"执行视觉检测 (第 {loop_count} 次), 当前物理坐标: {camera_coords}, 当前loading动作: {loading}, 当前拍照触发类型: {photo_type}")

            # 1. 调用相机接口
            result = self.take_photo_position(camera_coords, config, loading=loading, photo_type=photo_type)
            logger.info(f"photo result is >>>>>>>>: {result}")

            # 2. 解析结果
            res_status = result.get("res", "ng")
            trigger = result.get("trigger", "")
            coords = result.get("coords", [])

            # === 情况 A: 视觉 NG ===
            if res_status != "ok":
                logger.error(f"视觉返回 NG: {res_status}")
                return False

            """逻辑变动，trigger == "photo"，需要多次移动的情况暂时注释掉"""
            # === 情况 B: 需要移动重拍 (trigger == photo) ===
            # 逻辑：移动到 p_r0，然后 continue 继续下一轮拍照
            if trigger == "photo":
                if len(coords) < 1:
                    logger.error("视觉请求重拍，但未返回 p_r0 坐标")
                    return False

                p_r0_coords = coords[0]
                logger.info(f"工件过长/未拍全，移动到新拍照点: {p_r0_coords}")

                # 构造移动点
                next_photo_pt = {
                    "name": f"Vision_RePhoto_{loop_count}",
                    "coords": p_r0_coords,
                    "photo": 0  # 移动到位后，通过 while 循环再次触发 take_photo_position
                }

                # 执行移动 (Action 内部的移动)
                if not self._move_segment_to_target(process_addr=process_addr, target_point=next_photo_pt):
                    return False

                # 移动成功，更新末端记录，继续循环
                self.last_motion_end_point = next_photo_pt

                # 更新 camera_coords 为当前的新位置
                camera_coords = p_r0_coords

                continue

            # === 情况 C: 确认抓取 (trigger == retrieve) ===
            # 逻辑：保存 [p_r0, p_r1]，直接返回 True，把抓取留给动作 77
            elif trigger == "retrieve":
                if len(coords) < 2:
                    logger.error("视觉返回抓取，但坐标数量不足2个 (需 p_r0, p_r1)")
                    return False

                p_r0_coords = coords[0]  # 修正点/过渡点
                p_r1_coords = coords[1]  # 抓取点

                logger.info(f"视觉定位成功(Retrieve)，保存坐标供下一步使用: {coords}")

                # 只保存，不移动
                # eg：保存到 0x4008C (动作76) 的名下，供 77 读取
                self.save_vision_data(process_addr, coords, photo_type=photo_type)

                return True  # 动作 76 任务完成

            else:
                logger.error(f"未知的 trigger 类型: {trigger}")
                return False

        logger.error("视觉重拍次数过多，强制停止")
        return False

    def _handle_wood_stick_placement(self, layer_idx=None):
        """
        处理下料跨层放垫木逻辑：
        发送报警 -> 触发 UI -> 阻塞死等复位 -> 恢复
        """
        logger.warning("下料区检测到新的一层 (index=0)，挂起等待人工放置垫木！")

        # 1. 触发 UI 非阻塞弹窗 (你可以在界面新增一个类似的信号弹窗)
        self.sig_unloading_wood_stick_alarm.emit()

        # 2. 通知 PLC 亮起报警灯 (写入 14)
        logger.info(
            f"发送放垫木提示: {hex(const.ADDR_PRODUCT_UNLOADING_RACK)} -> {const.product_unloading_rack_wood_stick}")
        self.plc.write_register(const.ADDR_PRODUCT_UNLOADING_RACK, const.product_unloading_rack_wood_stick)

        # 3. 阻塞等待人工复位 (死等 10)
        logger.info("系统挂起：等待工人放置垫木，并按下机台复位按钮(10)...")

        if self.wait_for_plc_val(const.ADDR_PRODUCT_UNLOADING_RACK_RESET, const.product_unloading_rack_reset,
                                 timeout=-1):
            logger.info("收到放垫木复位信号(10)，发送 ACK(11)...")
            self.plc.write_register(const.ADDR_PRODUCT_UNLOADING_RACK_RESET, const.product_unloading_rack_reset_ack)

            # 报警解除，恢复正常状态 0
            self.plc.write_register(const.ADDR_PRODUCT_UNLOADING_RACK, 0)

            self.sig_unloading_wood_stick_clear.emit()
            logger.info("垫木放置完成，恢复自动下料！")
            return True

        return False  # 触发急停

    def handle_vision_recursive_v1(self, process_addr, point, loading=None, photo_type=const.photo_type_normal):
        """
        拍照识别逻辑
        料头、上料、下料的识别坐标，会保存到json文件，供get_vision_data调用

        :param process_addr: 动作地址位
        :param point: 拍照坐标
        :param loading, 上下料参数，1/上料，2/下料
        :param photo_type, 拍照触发的动作类型，普通拍照(物料识别)/1，上料(空料判断)/2，下料(满料判断)/3，铝屑识别/4, 默认1

        """
        loop_count = 0
        max_loops = 5

        point_coords = point.get("coords", [])
        config = point.get("config", "elbow_up")
        loading = loading

        while self.running and loop_count < max_loops:
            loop_count += 1
            realtime_pt = self.get_realtime_point()
            realtime_pt_coords = realtime_pt.get("coords", [])
            logger.info(
                f"执行视觉检测 (第 {loop_count} 次), 当前物理坐标: {realtime_pt_coords}, 当前loading动作: {loading}")

            # 1. 调用相机接口
            result = self.take_photo_position(point_coords, config, loading=loading, ptype=photo_type)
            logger.info(f"photo result is >>>>>>>>: {result}")

            # 2. 解析结果
            res_status = result.get("res", "error")
            coords = result.get("coords", [])
            layer = result.get("layer", -1)
            item_index = result.get("position", -1)

            # === 状态分支处理 ===
            if res_status == "error":
                logger.error(f"视觉返回 硬件/算法异常 (ERROR)")
                return "ERROR"
            elif res_status == "empty":
                logger.info("视觉检测完成：该位置无物料 (EMPTY)")
                return "EMPTY"

            # 到这里说明 res_status == "ok"
            logger.info(f"视觉执行成功, 返回原始坐标 {coords}")

            # # =======================================================
            # # 【核心新增】：下料区放垫木判定拦截
            # # =======================================================
            # if photo_type == const.photo_type_unloading and item_index == 0:
            #     physical_layer = (const.product_total_layers - 1) - layer
            #     if physical_layer == 0: # 第0层，料架自带垫木了，不需要提示
            #         pass
            #     else:
            #         # 阻塞在这里，直到工人放好木条并复位
            #         if not self._handle_wood_stick_placement():
            #             # 如果返回 False，说明被急停打断，直接退出
            #             return False

            transform_coords = []
            for coord in coords:
                if photo_type == const.photo_type_find_head:
                    # 找端头模式, 使用【相机】对齐目标
                    # 传入 cheat_gripper_offset 标志 (需在 transform_tool_coord 内部实现，让夹爪offset = 相机offset)
                    trans_coord = self.transform_tool_coord(coord, align_camera=1, joint_valid=False)
                elif photo_type == const.photo_type_unloading:
                    # 下料拍照，不做转换
                    trans_coord = coord
                else:
                    # 普通模式，使用【夹爪】对齐目标
                    trans_coord = self.transform_tool_coord(coord)

                if not trans_coord:
                    logger.error(f"工具坐标转换失败")
                    return "ERROR"
                transform_coords.append(trans_coord)

            # 保存转换后的视觉坐标，供运动动作读取
            # eg：保存到 0x4008C (动作76) 的名下，供 77 读取
            if photo_type in (const.photo_type_loading, const.photo_type_find_head, const.photo_type_unloading):
                self.save_vision_data(process_addr, transform_coords, photo_type=photo_type, layer=layer,
                                      index=item_index)

            return "OK"  # eg: 动作 76 任务完成

        logger.error("视觉重拍次数过多，强制停止")
        return "ERROR"

    def _wait_for_udp_response(self, sock, expected_msg="OK", timeout_sec=30.0):
        """
        带急停检测的 UDP 阻塞等待
        :param sock: UDP Socket 对象
        :param expected_msg: 期望收到的消息字符串
        :param timeout_sec: 总体超时时间（秒）
        :return: True(成功收到), False(超时、急停或异常)
        """
        # 设置底层 socket 超时时间为 0.5 秒，以保证能高频检查急停
        sock.settimeout(0.5)
        start_time = time.time()

        while self.running:
            # 1. 优先检查急停
            if self.check_estop():
                logger.warning("UDP 等待期间触发急停，终止等待")
                return False

            # 2. 检查总体超时
            if time.time() - start_time > timeout_sec:
                logger.error(f"等待 UDP 响应超时 (超过 {timeout_sec} 秒)")
                return False

            # 3. 尝试接收数据
            try:
                # 接收来自服务端的回复 (缓冲区 1024 字节足够了)
                data, addr = sock.recvfrom(1024)
                msg = data.decode('utf-8').strip()

                if msg == expected_msg:
                    return True
                else:
                    # 如果收到其他消息，可以忽略继续等，或者视为报错（看业务需求）
                    logger.warning(f"收到非预期的 UDP 消息: {msg}，继续等待 '{expected_msg}'")

            except socket.timeout:
                # socket 0.5秒超时是正常的，直接 continue 进入下一轮循环，去检查急停
                continue
            except Exception as e:
                logger.error(f"UDP 接收数据异常: {e}")
                return False

        return False

    def execute_standard_motion_sequence_with_interpolate(self, process_addr, points_sequence):
        """
        通用的运动控制序列执行函数，带笛卡尔插值计算
        :param points_sequence: 包含起点的完整点位列表 [Start, P1, P2...]
        """
        points_count = len(points_sequence)

        # 循环发送坐标
        for i in range(points_count - 1):
            # 1. 获取起点、终点坐标
            start_point = points_sequence[i]
            end_point = points_sequence[i + 1]

            # 2. 生成插值路径
            interpolated_path = TrajectoryV2().generate_cartesian_interpolated_path(
                [start_point, end_point],
                num_inserts=const.point_interpolated_num
            )
            points_to_send = interpolated_path[1:]  # 去掉起点

            # 拍照标志, 0/默认值，1/拍照，2/给坐标
            photo_trigger = end_point.get("photo", 0)
            target_name = end_point.get("name", f"P{i + 1}")

            # === while 重试循环 (你原来的逻辑) ===
            while self.running:
                logger.info(f"--- 执行段 {i + 1}/{points_count - 1}: {start_point.get('name')} -> {target_name} ---")
                # 3. 发送数据
                if not self.send_coords_batch(process_addr, points_to_send, 8):
                    logger.error("坐标发送失败，终止流程")
                    return False

                # 4. 握手
                self.plc.write_register(process_addr, 11)
                logger.info(f"坐标发送成功，发送响应11")

                # 5. 等待到位
                if not self.wait_for_plc_val(process_addr, 12, timeout=7200):
                    logger.error("等待机械臂到位超时")
                    return False

                logger.info(f"段 {target_name} 到位, plc回复 12)")

                # 拍照/定位处理
                vision_ok = True

                if photo_trigger == 1:  # 检查
                    # if self.take_photo_check() == "NG": vision_ok = False
                    logger.info("触发拍照...")
                    photo_res = self.take_photo()  # 返回 "OK" 或 "NG"

                    if photo_res == "OK":
                        logger.info("拍照结果: OK")
                        # 拍照成功，发送 15 (根据协议，可能需要告知PLC拍照OK)
                        # 注意：如果不是最后一步，发送15可能会覆盖状态，需确认协议细节。
                        # 通常做法：NG才报错，OK则继续。这里假设OK需要发15确认。 暂时注释掉
                        # self.plc.write_register(process_addr, 15)
                        vision_ok = True
                    else:
                        logger.error("拍照结果: NG")
                        vision_ok = False
                elif photo_trigger == 2:  # 定位
                    coords_list = self.take_photo_position()
                    if coords_list and len(coords_list) > 0:
                        logger.info(f"视觉定位成功，获取到 {len(coords_list)} 个目标")
                        # [保存数据] Key 使用当前的 process_addr
                        self.save_vision_data(process_addr, coords_list, photo_type=const.photo_type_loading)
                        # self.vision_target_coords = pos  # 保存
                    else:
                        logger.error("视觉定位失败 (未识别到目标)")
                        vision_ok = False

                # 结果分支
                if vision_ok:
                    break
                else:
                    self.plc.write_register(process_addr, 16)
                    # 等待 20 复位...
                    if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                        continue
                    else:
                        return False

        # 全部完成
        self.plc.write_register(process_addr, 13)
        logger.info("所有点位执行完毕，发送完成信号 13")
        return True

    def execute_standard_motion_sequence(self, process_addr, points_sequence, loading=None, photo_type=None,
                                         send_done=True, vision_retry=False):
        """
        标准运动序列执行函数, 没有插值
        :param process_addr:动作地址位
        :param points_sequence: 坐标点位
        :param loading: 上下料标记，1/上料，2/下料
        :param photo_type: 普通拍照(物料识别)/1，上料(空料判断)/2，下料(满料判断)/3，铝屑识别/4
        :param send_done: 执行完毕后是否向 PLC 发送 13 完成信号。默认为 True。
        :param vision_retry, 缝激活“死等20并重试”的逻辑
        """
        # 1. 检查急停
        if self.check_estop():
            logger.warning("当前处于急停状态，拒绝执行新任务")
            return False

        points_count = len(points_sequence)

        # udp连接
        udp_client = UdpClient(
            local_port=const.inspection_udp_local_port,
            remote_ip=const.inspection_udp_ip,
            remote_port=const.inspection_udp_port
        )

        has_ccd_triggered = False  # 记录本轮流程是否触发过 CCD
        has_laser_triggered = False  # 记录本轮流程是否触发过 Laser
        has_depth_vision_triggered = False

        aluminum_exists = False  # 铝屑识别的时候，默认没有铝屑

        ccd_ok = True
        laser_ok = True

        try:
            for i in range(points_count - 1):
                start_point = points_sequence[i]
                end_point = points_sequence[i + 1]
                logger.info(f"target point : {end_point}")

                # 获取 photo 标志 (0 或 1)
                photo_trigger = end_point.get("photo", 0)

                # === while 重试循环 (处理 NG -> 16 -> 20 -> 重试) ===
                while self.running:
                    # 1. 检查急停
                    if self.check_estop():
                        logger.critical("流程强制终止：急停触发")
                        self.last_motion_end_point = None  # 清除记忆点，强制下次重新获取实时位置
                        return False  # 直接退出函数，即清除了当前流程数据

                    # 1. 执行移动 (使用提取出的通用函数)
                    # 这会处理插值、发送、等待12
                    if not self._move_segment_to_target(process_addr=process_addr, start_point=start_point,
                                                        target_point=end_point):
                        target_name = end_point.get("name", "Temp_Point")
                        logger.error(f"移动到 {target_name} 失败，流程异常终止")
                        return False  # 移动失败(如急停)，直接退出

                    # 2. 拍照逻辑处理
                    vision_ok = True

                    if photo_trigger == const.photo_trigger_depth:
                        has_depth_vision_triggered = True

                        ###########################################
                        # 测试代码，工装气缸夹爪测试，算法未完善，跳过
                        ###########################################
                        if photo_type == const.photo_type_cylinder:
                            # vision_res = self.handle_vision_recursive_v1(process_addr, end_point, loading, const.photo_type_cylinder)
                            vision_ok = True
                        else:

                            # 调用视觉逻辑
                            vision_res = self.handle_vision_recursive_v1(process_addr, end_point, loading, photo_type)
                            if vision_res == "OK":
                                vision_ok = True
                            else:
                                # EMPTY 或者 ERROR
                                vision_ok = False

                                # 判断是否是铝屑识别，有铝屑
                                if photo_type == const.photo_type_aluminum:
                                    aluminum_exists = True

                    elif photo_trigger == const.photo_trigger_ccd:
                        # CCD 相机触发逻辑
                        pos_name = end_point.get("name", "UnknownPos")
                        coords = end_point.get("coords", [0.0, 0.0, 0.0, 0.0])
                        x, y, z = coords[0], coords[1], coords[2]

                        # 组装字符串格式：ccd_pos_x_y_z (保留2位小数防数据过长)
                        msg = f"ccd_{pos_name}_{x:.2f}_{y:.2f}_{z:.2f}"
                        udp_client.send_msg(msg)
                        logger.info(f"发送 UDP (CCD): {msg}")
                        has_ccd_triggered = True

                        time.sleep(1)
                        # 阻塞等待 OK，超时时间设为 60 秒 (根据实际算法耗时调整)
                        # 调用封装的等待方法，把 controller 的急停检测方法当做参数传进去

                        # if udp_client.wait_for_response(
                        #         expected_msg="OK",
                        #         timeout_sec=60.0,
                        #         check_estop_func=self.check_estop,  # 注入急停检测回调
                        #         is_running_func=lambda: self.running  # 注入线程状态回调
                        # ):
                        #     logger.info(f"[{pos_name}] 收到 CCD 响应: OK")
                        #     ccd_ok = True
                        # else:
                        #     logger.error(f"[{pos_name}] CCD 响应失败或超时")
                        #     ccd_ok = False
                        #
                        # ccd_ok = False
                    elif photo_trigger == const.photo_trigger_laser:
                        # 激光测距触发逻辑
                        pos_name = end_point.get("name", "UnknownPos")
                        coords = end_point.get("coords", [0.0, 0.0, 0.0, 0.0])
                        x, y, z = coords[0], coords[1], coords[2]

                        # 组装字符串格式：laser_pos_x_y_z
                        msg = f"laser_{pos_name}_{x:.2f}_{y:.2f}_{z:.2f}"
                        udp_client.send_msg(msg)
                        logger.info(f"发送 UDP (Laser): {msg}")
                        has_laser_triggered = True

                        time.sleep(1)
                        # if udp_client.wait_for_response(
                        #         expected_msg="OK",
                        #         timeout_sec=30.0,
                        #         check_estop_func=self.check_estop,
                        #         is_running_func=lambda: self.running
                        # ):
                        #     logger.info(f"[{pos_name}] 收到 Laser 响应: OK")
                        #     laser_ok = True
                        # else:
                        #     logger.error(f"[{pos_name}] Laser 响应失败或超时")
                        #     laser_ok = False
                        # laser_ok = False
                    # 3. 结果分支 (NG 处理)
                    if vision_ok:
                        break  # 成功，退出 while，进入下一个 for (如果有的话)
                    else:
                        logger.error("视觉/外设处理结果为 NG，发送 16")
                        if send_done:
                            self.plc.write_register(process_addr, 16)

                        if vision_retry:
                            logger.warning("开启了重试模式：等待复位 (20)...")
                            # 等待 PLC 复位信号
                            if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                                logger.info("收到 20，重试当前步骤")
                                # 这里的 continue 会导致重新执行 _move_segment_to_target
                                # 也就是重新走到 end_point，然后重新触发拍照
                                continue  # 回到 while 循环头部重试
                            else:
                                return False
                        else:
                            logger.info("未开启重试模式：交由 PLC 控制逻辑跳转，流程结束")
                            return False  # 直接退出当前引擎

            # ========================================================
            # 所有点位执行完成，发送UDP结束信号
            # ========================================================
            if has_ccd_triggered:
                udp_client.send_msg("ccd_finished")
                logger.info("发送 UDP: ccd_finished")

            if has_laser_triggered:
                udp_client.send_msg("laser_finished")
                logger.info("发送 UDP: laser_finished")

            # 序列全部完成，发送 13
            if send_done:
                # 调用深度相机，发送15
                if has_depth_vision_triggered:
                    if aluminum_exists:  # 如果识别到铝屑，在上面的vision=False分支，已经发送了16，不再发送15
                        pass
                    else:
                        logger.info(f"标准序列全部完成 (执行过 Vision 拍照分析)，发送 15 结束信号")
                        self.plc.write_register(process_addr, 15)
                else:  # 调用ccd相机，或者激光相机，另外处理
                    if not ccd_ok or not laser_ok:  # 如果 ccd检测或laser检测有一个失败，发送16
                        logger.info(f"标准序列全部完成 (执行ccd 或 激光 拍照分析)，失败，发送 16 结束信号")
                        self.plc.write_register(process_addr, 16)
                    else:
                        logger.info(f"标准序列全部完成 (执行ccd 或 激光 拍照分析)，成功，发送 13 结束信号")
                        self.plc.write_register(process_addr, 13)
            else:
                logger.info(f"标准序列完成，等待后续拼接动作 (暂不发 13)")

            return True

        finally:
            # 强制释放，保证端口绝对不会被占用。
            udp_client.close()

    def _handle_empty_rack_and_wait(self, process_addr):
        """
        处理上料料架全空逻辑：
        终止当前动作 -> 发送缺料报警 -> 阻塞死等人工换料 -> 回复复位ACK
        """
        logger.critical("料架全空！请更换料车！")

        #  优先移动到空料安全点，便于新的料车上料
        try:
            # 1. 读取 empty_points 配置
            empty_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("empty_points", [])

            if empty_points:
                logger.info(f"发现空料安全点配置，开始安全撤离 (共 {len(empty_points)} 个点)...")

                # 2. 获取当前实时坐标作为起点，确保轨迹连续
                realtime_pt = self.get_realtime_point()
                if realtime_pt:
                    safe_sequence = [realtime_pt] + empty_points

                    # 3. 执行撤离运动
                    # 必须设置 send_done=False，因为此时动作还没结束，接下来还要发 13 和报警
                    move_ok = self.execute_standard_motion_sequence(
                        process_addr=process_addr,
                        points_sequence=safe_sequence,
                        send_done=False
                    )

                    if not move_ok:
                        logger.error("移动到空料安全点失败或被急停打断！")
                else:
                    logger.error("获取实时坐标失败，取消安全撤离。")
            else:
                logger.warning("未配置空料安全点 (empty_points)，机械臂将停在原地报警。")

        except Exception as e:
            logger.error(f"执行空料安全点撤离时发生异常: {e}")

        # 1. 内部状态重置
        self.loading_index.reset_search_index(const.search_index_name)  # 自动复位，准备迎接新料车
        self.loading_index.reset_search_index(const.head_layer_index_name)  # 自动复位，准备迎接新料车
        self.loading_index.reset_search_index(const.last_picked_layer_name, -1)

        self.loading_index.current_search_index = 0
        self.loading_index.current_head_layer_index = 0
        self.loading_index.last_picked_layer = -1

        self.need_rack_mapping = True  # 要求换料车后重新扫描

        # 2. 终止当前业务动作
        # 告诉 PLC 当前寻找动作完成 (这里你用了13，如果和PLC约定的没找到也发13就没问题；如果约定没找到发16，就改成16)
        self.plc.write_register(process_addr, 13)

        # 3. 通知 PLC 全局缺料
        logger.info(f"发送料架空报警: {hex(const.ADDR_PRODUCT_LOADING_RACK)} -> {const.product_loading_rack_empty}")
        self.plc.write_register(const.ADDR_PRODUCT_LOADING_RACK, const.product_loading_rack_empty)

        # 4. 阻塞等待人工换料并复位
        logger.info("系统挂起：等待工人更换料车并按下复位按钮...")

        # 无限等待复位信号 (10)
        if self.wait_for_plc_val(const.ADDR_PRODUCT_LOADING_RACK_RESET, const.product_loading_rack_reset, timeout=-1):
            logger.info("检测到新料车复位信号(10)，发送确认收到(11)...")
            self.plc.write_register(const.ADDR_PRODUCT_LOADING_RACK_RESET, const.product_loading_rack_reset_ack)

            self.plc.write_register(const.ADDR_PRODUCT_LOADING_RACK, 0)

            logger.info("工人已更换料车并复位，流程重新开始")
            # 返回 False，退出当前调用栈，让主循环重新接收 PLC 的新指令
            return False

        return False

    # 处理取垫木挂起逻辑的辅助函数
    def _handle_wood_stick_removal(self, process_addr, layer_idx):
        """
        处理跨层取垫木逻辑：
        发送报警 -> 触发 UI -> 阻塞死等复位 -> 恢复
        """
        logger.warning(f"检测到跨层 (目标层：{layer_idx})，挂起等待人工移除垫木！")

        # 1. 触发 UI 非阻塞弹窗
        self.sig_wood_stick_alarm.emit(layer_idx)

        # 2. 发送状态 (13: 垫木)
        self.plc.write_register(const.ADDR_PRODUCT_LOADING_RACK, const.product_loading_rack_wood_stick)

        # 3. 阻塞等待人工复位 (按照约定，等待 PLC 在当前动作地址发送 20)
        logger.info("系统挂起：等待工人取走垫木，并按下机台复位按钮(20)...")

        # 使用 timeout=-1 无限死等，期间 check_estop 依然有效
        if self.wait_for_plc_val(const.ADDR_PRODUCT_LOADING_RACK_RESET, const.product_loading_rack_reset, timeout=-1):
            logger.info("收到垫木移除复位信号(10)，恢复自动运行！")

            # (可选) 恢复全局料架状态为正常，发送ack 11
            self.plc.write_register(const.ADDR_PRODUCT_LOADING_RACK_RESET, const.product_loading_rack_reset_ack)

            # 4. 将全局状态地址归零，表示报警解除，恢复正常
            self.plc.write_register(const.ADDR_PRODUCT_LOADING_RACK, 0)

            # 通知 UI 关闭弹窗
            self.sig_wood_stick_clear.emit()

            return True

        return False  # 触发急停或程序退出

    def scan_rack_depth_mapping(self, process_addr, search_points, config, loading, photo_type):
        """
        料架顶层扫描映射 (记忆跃进版)
        利用记忆索引直接跳过已清空的上方楼层，极大提升开机恢复效率
        """
        logger.info("========================================")
        logger.info("开始执行料架映射扫描 (Smart Pallet Mapping)...")

        COLS_PER_LAYER = const.product_per_layer
        # 每一层的固定落差
        LAYER_GAP = const.product_height + const.interval_height  # 177.0 mm (物料 + 木条)
        # 基础高度，相机到物料的拍摄距离
        BASE_DEPTH = const.base_depth
        # 可信深度，需要加上容差
        MAX_RELIABLE_DEPTH = BASE_DEPTH + LAYER_GAP + const.depth_tolerange

        # 总层数
        total_points = len(search_points)
        total_layers = math.ceil(total_points / COLS_PER_LAYER)

        # ========================================================
        # 【核心优化】：计算起始扫描层
        # ========================================================
        # 假设当前索引是 12，那么 12 // 5 = 2，直接从第 2 层开始扫
        # 如果是新料车（索引被重置为0），0 // 5 = 0，从第 0 层开始扫
        current_idx = self.loading_index.current_search_index
        start_layer_idx = current_idx // COLS_PER_LAYER

        # 容错：防止由于配置变动导致层数越界
        start_layer_idx = max(0, min(start_layer_idx, total_layers - 1))

        logger.info(f"根据记忆索引 [{current_idx}]，智能跳过上方空层，直接从第 {start_layer_idx} 层开始扫描")
        logger.info("========================================")

        # === 外层循环：从 start_layer_idx 开始逐层下探 ===
        for layer_idx in range(start_layer_idx, total_layers):
            if self.check_estop(): return -1

            logger.info(f"--- 开始扫描第 {layer_idx} 层高度视野 ---")

            # 提取当前层需要扫描的 5 个点
            start_idx = layer_idx * COLS_PER_LAYER
            end_idx = min(start_idx + COLS_PER_LAYER, total_points)
            current_layer_points = search_points[start_idx:end_idx]

            col_depths = []

            # === 内层循环：扫描该层的 5 列 ===
            for col, pt in enumerate(current_layer_points):
                if self.check_estop(): return -1

                logger.info(f"映射扫描: 前往 {pt['name']} ...")

                # 移动并拍照...
                if not self._move_segment_to_target(process_addr, target_point=pt):
                    return -1

                result = self.take_photo_position(pt.get("coords"), config, loading, photo_type)
                res_status = result.get("res", "ng")
                coords = result.get("coords", [])

                depth = float('inf')
                if res_status == "ok" and coords:
                    depth = coords[0][2]
                    logger.info(f"  -> 第 {col + 1} 列检测到物料，深度 Zc = {depth:.1f} mm")
                else:
                    logger.info(f"  -> 第 {col + 1} 列视野为空")

                col_depths.append(depth)

            # === 分析当前层的扫描结果 ===
            min_depth = min(col_depths)

            if min_depth <= MAX_RELIABLE_DEPTH:
                valid_items = []
                for col, depth in enumerate(col_depths):
                    if depth <= MAX_RELIABLE_DEPTH:
                        # 相对层数计算
                        relative_layer = round((depth - BASE_DEPTH) / LAYER_GAP)

                        # 绝对层数 = 当前扫描层 + 相对层数
                        absolute_layer = layer_idx + relative_layer
                        absolute_layer = max(0, min(absolute_layer, total_layers - 1))
                        valid_items.append((col, absolute_layer, depth))

                if valid_items:
                    min_abs_layer = min(item[1] for item in valid_items)
                    items_in_top_layer = [item for item in valid_items if item[1] == min_abs_layer]

                    best_item = items_in_top_layer[0]
                    best_col = best_item[0]
                    best_abs_layer = best_item[1]
                    best_depth = best_item[2]

                    target_index = best_abs_layer * COLS_PER_LAYER + best_col

                    logger.info("========================================")
                    logger.info(f"映射成功！锁定目标：第 {best_abs_layer} 层，第 {best_col} 列")
                    logger.info(f"推算总列表索引为 [{target_index}]")
                    logger.info("========================================")

                    return target_index

            else:
                logger.warning(f"第 {layer_idx} 层扫描未发现近距离物料，准备下探...")

        logger.critical("映射失败：所有高度层扫描完毕，料架已完全空载！")
        return -1

    def execute_search_motion_sequence(self, process_addr, search_points, loading=None, photo_type=None):
        """
        阵列搜寻专用执行引擎
        逻辑：按记忆索引顺序移动，拍到空料则跳过，拍到物料则保存并结束。全空则报警。
        :param process_addr:动作地址位
        :param points_sequence: 坐标点位
        :param loading: 上下料标记，1/上料，2/下料
        :param photo_type, 普通拍照(物料识别)/1，上料(空料判断)/2，下料(满料判断)/3，铝屑识别/4
        """
        if self.check_estop(): return False

        if self.loading_index.current_search_index >= len(search_points):
            logger.warning("搜寻索引已越界，自动重置为 0 (可能是新料架)")
            self.loading_index.reset_search_index(const.reset_search_index)

        # ==========================================
        # 1. 初始化扫描判定
        # ==========================================
        # 如果是新料车，或者刚开机，强制执行顶部扫描来纠正 index
        if getattr(self, 'need_rack_mapping', True):
            # 提取公共 config
            config = search_points[0].get("config", "elbow_up") if search_points else "elbow_up"
            target_idx = self.scan_rack_depth_mapping(process_addr, search_points, config, loading, photo_type)

            if target_idx == -1:
                # 料架全空或异常，通知plc，plc执行动作(停止上料)
                return self._handle_empty_rack_and_wait(process_addr)

            # 扫描成功，覆盖当前的搜寻索引
            self.loading_index.save_search_index(const.search_index_name, target_idx)
            # 关闭映射标志，后续的动作直接按照索引往下抓即可
            self.need_rack_mapping = False

        # ==========================================
        # 2. 从确定的索引开始循环搜寻
        # ==========================================

        for i in range(self.loading_index.current_search_index, len(search_points)):
            if self.check_estop(): return False

            target_point = search_points[i]
            logger.info(f">>> 开始搜寻点位[{i + 1}/{len(search_points)}]: {target_point.get('name')} <<<")

            # ==== 重试大循环 (应对相机故障) ====
            while self.running:
                # 急停监听
                if self.check_estop(): return False

                # ========================================================
                # 姿态切换检测与【多点安全过渡】逻辑
                # ========================================================
                realtime_pt = self.get_realtime_point()

                curr_config = realtime_pt.get("config", "elbow_up") if realtime_pt else "elbow_up"
                target_config = target_point.get("config", "elbow_up")

                # # 如果检测到接下来的目标点需要翻肘
                # if curr_config != target_config:
                #     logger.warning(f"检测到机械臂姿态即将切换 ({curr_config} -> {target_config})")
                #     # 读取配置中的多点安全过渡序列
                #     flip_via_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get(
                #         "flip_via_points",[])
                #
                #     if flip_via_points:
                #         logger.info(f"开始执行姿态切换安全过渡序列 (共 {len(flip_via_points)} 个点)...")
                #
                #         # for 循环直接调用 _move_segment_to_target(..., interpolate=False)
                #         for fp in flip_via_points:
                #             if not self._move_segment_to_target(process_addr, target_point=fp, interpolate=False):
                #                 return False
                #
                #         logger.info("姿态安全过渡执行完毕，准备前往最终目标点")

                # 1. 移动到端头搜寻点
                if not self._move_segment_to_target(process_addr=process_addr, target_point=target_point):
                    return False

                # 2. 触发端头拍照识别 (获取新定义的三态返回值)
                vision_res = self.handle_vision_recursive_v1(process_addr, target_point, loading, photo_type)

                # # 测试移动
                # if i == 0:
                #     vision_res = "EMPTY"

                # 3. 结果分支处理
                if vision_res == "OK":
                    # 找到了！保存进度，下次还从这个视野开始找 (因为一个视野可能有两个料)
                    # 或者如果一个视野只抓一次，可以存 i + 1。这里假设存 i
                    self.loading_index.save_search_index(const.search_index_name, i)

                    logger.info("=========================================")
                    logger.info("端头已找到！准备平移 1020mm 进行精确定位")
                    logger.info("=========================================")

                    # 1. 获取刚才转换好的“端头法兰绝对坐标”
                    p_head_data = self.get_vision_data(process_addr, photo_type=const.photo_type_find_head)
                    if not p_head_data:
                        return False

                    x_f, y_f, z_f, r_f = p_head_data[0]

                    # 2. 计算精拍点的绝对坐标 (直接 Y + 1020)
                    target_y = y_f + const.product_y_offset

                    # 3. 根据定义的末端绝对角度，计算j4
                    x_f_p = x_f
                    y_f_p = target_y
                    # 取原本悬停的安全 Z 高度
                    z_f_p = target_point["coords"][2]
                    config_f_p = target_point.get("config", "elbow_up")

                    j4 = ScaraKinematics().calculate_motor_r_from_world_angle(
                        x_f_p, y_f_p, z_f_p, const.fine_photo_world_angle,
                        self.l1, self.l2, self.z0, self.nn3,
                        elbow_config=config_f_p)

                    # ############################
                    # # 智能计算j4, 和elbow位姿
                    # ############################
                    # _ = self.get_realtime_point()
                    # current_j2 = self.last_joint_status[1]
                    #
                    # # === 4. 调用智能函数，同时获取 J4 和 安全的 Config ===
                    # j4, safe_config = ScaraKinematics.calculate_motor_r_from_world_angle_smart(
                    #     x_f_p, y_f_p, z_f_p, const.fine_photo_world_angle,
                    #     self.l1, self.l2, self.z0, self.nn3,
                    #     current_j2=current_j2  # 传入当前 J2
                    # )
                    # config_f_p = safe_config

                    if j4 is None:
                        logger.error("计算精拍点 J4 补偿角度失败，目标可能超限")
                        return False

                    # 4. 构造精拍点对象
                    fine_photo_pt = {
                        "name": "Fine_Photo_Pos_1020",
                        "coords": [x_f_p, y_f_p, z_f_p, j4],
                        # 注意 Z 和 R 保持最初的安全悬停高度和姿态
                        "config": config_f_p,
                        "photo": 0
                    }

                    logger.info(f"端头法兰Y: {y_f:.2f}, 平移后精拍法兰Y: {target_y:.2f}")

                    # 4. 移动到精拍点
                    if not self._move_segment_to_target(process_addr, target_point=fine_photo_pt):
                        return False

                    # 5. 第 2 拍：精确定位 (这次用 photo_type_loading，它会算出夹爪的真实抓取坐标)
                    final_vision_res = self.handle_vision_recursive_v1(process_addr, fine_photo_pt, loading,
                                                                       photo_type=const.photo_type_loading)

                    if final_vision_res == "OK":
                        logger.info("精确定位成功！最终抓取坐标已保存。")
                        self.last_motion_end_point = fine_photo_pt
                        self.plc.write_register(process_addr, 13)
                        return True
                    else:
                        # 精拍失败，发 16 报警等人工处理
                        logger.info("精确定失败！")
                        self.plc.write_register(process_addr, 16)
                        if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                            continue  # 收到 20 后，重头开始找端头
                        return False

                elif vision_res == "EMPTY":
                    # 没找到，属于正常现象！
                    logger.info(f"该位置无料，准备前往下一个搜寻点...")
                    # 跳出 while 重试循环，继续 for 循环走下一个点
                    break

                else:  # "ERROR"
                    # 相机真报错了 (比如线断了)，按照原逻辑发 16 报警
                    self.plc.write_register(process_addr, 16)
                    logger.warning("视觉严重异常 (16)，等待人工复位 (20)...")
                    if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                        logger.info("收到 20，重试当前步骤")
                        continue  # 回到 while 开头重新移动、重新拍照
                    else:
                        return False

        # ==========================================
        # 如果 for 循环正常结束，说明所有点都拍完了，全是 "EMPTY"
        # ==========================================
        return self._handle_empty_rack_and_wait(process_addr)

    def execute_two_stage_vision_sequence_option_2(self, process_addr, config_data, loading, photo_type):
        """
        双阶段视觉执行引擎 (端头群拍 -> 物理X轴映射 -> Y轴平移 -> 二次精拍)
        方案二
        用于解决大视野找端头带来的 X 轴畸变问题。

        此方案的特点在于利用原先设定好的精拍阵列搜寻点位
        端头找到物料坐标之后，利用转换后的X，计算近似的精拍搜索点位，运动到固定搜索点位拍照
        """
        if self.check_estop(): return False

        # 1. 提取配置
        # head_points = config_data.get("layer_head_points", [])
        # precise_points = config_data.get("layer_precise_points", [])

        head_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("layer_head_points", [])
        precise_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("search_points", [])

        if not head_points or not precise_points:
            logger.error("双阶段视觉配置缺失：未找到端头点或精拍预设点")
            return False

        # 从记忆层数开始 (如果你需要像之前那样跳过空层)
        # 这里假设你新增了一个 current_layer_index 变量来记忆
        # start_layer = getattr(self.loading_index, const.head_layer_index_name, 0)

        cols_per_layer = const.product_per_layer
        current_idx = self.loading_index.current_search_index
        start_layer = current_idx // cols_per_layer

        # 确保不越界
        start_layer = max(0, min(start_layer, len(head_points) - 1))

        # =======================================================
        # 阶段 1：逐层寻找端头
        # =======================================================
        found_layer_idx = -1
        valid_materials_base_coords = []
        target_material_base_coord = None  # 直接存放唯一的目标基座标

        for layer_idx in range(start_layer, len(head_points)):
            if self.check_estop(): return False

            head_pt = head_points[layer_idx]
            logger.info(f"--- [阶段1] 前往第 {layer_idx} 层端头群拍点: {head_pt.get('name')} ---")

            while self.running:
                if self.check_estop(): return False

                # 1. 移动到端头群拍点
                if not self._move_segment_to_target(process_addr=process_addr, target_point=head_pt):
                    return False

                # 2. 触发端头拍照 (注意 ptype: 找端头模式，视觉内部不要转换夹爪坐标)
                vision_res = self.handle_vision_recursive_v1(process_addr, head_pt, loading,
                                                             photo_type=const.photo_type_find_head)

                if vision_res == "OK":
                    # 找到了！读取这层所有端头的基座标 (假设视觉存入了缓存)
                    p_head_data = self.get_vision_data(process_addr, photo_type=const.photo_type_find_head)

                    # if p_head_data:
                    #     logger.info(f"第 {layer_idx} 层端头发现 {len(p_head_data)} 根物料！")
                    #     valid_materials_base_coords = p_head_data
                    #     found_layer_idx = layer_idx
                    #     self.loading_index.current_layer_index = layer_idx  # 保存层数记忆
                    #     break  # 跳出 while 重试循环
                    # else:
                    #     logger.error("视觉返回OK，但无法获取端头坐标数据")

                    if p_head_data and len(p_head_data.get("coords", [])) > 0:
                        logger.info(f"第 {layer_idx} 层端头发现物料！")
                        target_material_base_coord = p_head_data.get("coords")[0]
                        found_layer_idx = layer_idx

                        self.loading_index.current_head_layer_index = layer_idx  # 保存层数记忆
                        self.loading_index.save_search_index(const.head_layer_index_name, layer_idx)
                        break  # 跳出 while 重试循环
                    else:
                        logger.error("视觉返回OK，但无法获取端头坐标数据")
                        # 可以当作 ERROR 处理，发16等20

                elif vision_res == "EMPTY":
                    logger.info(f"第 {layer_idx} 层端头无料，准备下探...")
                    break  # 跳出 while，执行外层 for 的下一层

                else:  # "ERROR"
                    self.plc.write_register(process_addr, 16)
                    logger.warning("端头视觉异常 (16)，等待复位 (20)...")
                    if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                        continue
                    else:
                        return False

            # 如果在这一层找到了料，就跳出下探循环，进入精拍阶段
            if found_layer_idx != -1:
                break

        # 如果所有层都找完了还是没料
        if found_layer_idx == -1:
            return self._handle_empty_rack_and_wait(process_addr)

        # =======================================================
        # 阶段 2：X 轴映射与 Y 轴平移，生成精拍点位
        # =======================================================
        logger.info("=========================================")
        logger.info("端头已找到！开始进行 X 轴映射与 Y 轴平移")

        if not target_material_base_coord:
            return False

        # 提取当前层预设的 5 个精确卡槽点
        # cols_per_layer = const.product_per_layer
        start_idx = found_layer_idx * cols_per_layer
        end_idx = start_idx + cols_per_layer
        current_layer_precise_pts = precise_points[start_idx:end_idx]

        # 提取这 5 个预设点的基座标 X 值
        taught_x_list = [pt["coords"][0] for pt in current_layer_precise_pts]

        target_precise_points = []

        # for mat_base_coord in valid_materials_base_coords:
        # mat_x, mat_y, mat_z, mat_r = mat_base_coord
        mat_x, mat_y, mat_z, mat_r = target_material_base_coord

        # 【核心映射算法】：计算视觉 X 与所有 5 个预设 X 的距离，找最近的
        distances = [abs(mat_x - taught_x) for taught_x in taught_x_list]
        min_dist_idx = distances.index(min(distances))

        # 计算出这根物料在 35 个点中的全局绝对索引！
        absolute_item_index = start_idx + min_dist_idx

        matched_taught_pt = current_layer_precise_pts[min_dist_idx]

        logger.info(
            f"视觉端头 X={mat_x:.1f} -> 映射为槽位 P{min_dist_idx} (预设X={taught_x_list[min_dist_idx]:.1f})")

        # 计算精拍点的绝对坐标
        # Y = 视觉真实端头 Y + 1020mm
        target_y = mat_y + const.product_y_offset

        # X, Z 取预设卡槽的安全值
        target_x = matched_taught_pt["coords"][0]
        target_z = matched_taught_pt["coords"][2]
        config_curr = matched_taught_pt.get("config", "elbow_up")

        # 根据定义的末端绝对角度，计算 J4
        j4 = ScaraKinematics().calculate_motor_r_from_world_angle(
            target_x, target_y, target_z, const.fine_photo_world_angle,
            self.l1, self.l2, self.z0, self.nn3,
            elbow_config=config_curr
        )

        if j4 is None:
            logger.error(f"计算精拍点 J4 失败，槽位 P{min_dist_idx} 偏移后可能超限")
            return False

        fine_photo_pt = {
            "name": f"Fine_Photo_P{min_dist_idx}",
            "coords": [target_x, target_y, target_z, j4],
            "config": config_curr,
            "photo": 1  # 标记为需要触发精拍
        }
        # target_precise_points.append(fine_photo_pt)
        #
        # if not target_precise_points:
        #     logger.error("所有视觉端头映射后均无法生成合法的精拍点！")
        #     return False

        # =======================================================
        # 阶段 3：执行精拍与抓取
        # =======================================================
        # 为了不破坏原有单根抓取逻辑，我们只取第一个映射成功的点去精拍抓取
        # 下次请求时，它会重新拍端头，找出剩下的料

        # current_target = target_precise_points[0]
        current_target = fine_photo_pt
        logger.info(f"=== [阶段3] 前往精拍点: {current_target['name']} ===")

        while self.running:
            if self.check_estop(): return False

            # 1. 移动到精拍点
            if not self._move_segment_to_target(process_addr=process_addr, target_point=current_target):
                return False

            # 2. 在精拍点触发真实的单根物料抓取识别 (photo_type_loading)
            # 这次拍照会调用 transform_tool_coord 算出夹爪的真实抓取坐标并保存
            final_vision_res = self.handle_vision_recursive_v1(process_addr, current_target, loading,
                                                               photo_type=const.photo_type_loading)

            if final_vision_res == "OK":
                logger.info("精确定位成功！最终抓取坐标已保存。")
                # 保存抓取索引
                self.loading_index.current_search_index = absolute_item_index
                self.loading_index.save_search_index(const.search_index_name, absolute_item_index)

                self.last_motion_end_point = current_target
                self.plc.write_register(process_addr, 13)
                return True

            elif final_vision_res == "EMPTY":
                # 理论上精拍不该为空，因为端头看到了。如果为空可能是被碰掉了或光线变化
                logger.error("精拍发现位置为空，异常！")
                # 视为 ERROR 处理
                self.plc.write_register(process_addr, 16)
                if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                    # 收到 20 后，直接 return False 退出当前引擎，让外层/PLC重新发起整个找料流程
                    return False
                return False

            else:
                self.plc.write_register(process_addr, 16)
                logger.warning("精拍失败 (16)，等待复位 (20)...")
                if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                    continue
                return False

        return False

    def execute_two_stage_vision_sequence_option_1(self, process_addr, loading=None, photo_type=None):
        """
        上料区，料头拍摄+移动+料身精拍
        方案一
        执行引擎：端头群拍 -> 提取坐标 -> 纯视觉Y+平移 -> 二次精拍
        """
        if self.check_estop(): return False

        head_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("layer_head_points", [])

        if not head_points:
            logger.error("配置缺失：未找到端头拍照点 (layer_head_points)")
            return False

        # 1. 获取平移参数
        y_offset_dist = const.product_y_offset
        x_offset_dist = const.depth_interference_x_offset

        # 2. 读取记忆的层数 (假设你已经有了 current_layer_index 属性)，layer_idx从0开始
        start_layer = getattr(self.loading_index, 'current_head_layer_index', 0)
        start_layer = max(0, min(start_layer, len(head_points) - 1))

        # =======================================================
        # 阶段 1：逐层寻找端头
        # =======================================================
        found_layer_idx = -1
        target_material_base_coord = None  # 直接存放唯一的目标基座标

        total_layers = len(head_points)
        layer_idx = start_layer

        # === 外层循环：逐层寻找端头 ===
        while layer_idx < total_layers:
            if self.check_estop(): return False
            if not self.running: return False

            head_pt = head_points[layer_idx]
            logger.info(f"---> 前往第 {layer_idx} 层端头群拍点: {head_pt.get('name')} <---")

            # 1. 移动到端头拍照点
            if not self._move_segment_to_target(process_addr, target_point=head_pt):
                return False

            # 2. 触发端头群拍 (ptype=5: 找端头模式，不转换夹爪坐标)
            vision_res = self.handle_vision_recursive_v1(process_addr, head_pt, loading,
                                                         photo_type=const.photo_type_find_head)

            # ==========================================
            # 分支 1：找到物料 (OK)
            # ==========================================
            if vision_res == "OK":
                p_head_data = self.get_vision_data(process_addr, photo_type=const.photo_type_find_head)
                logger.info(f"head_data: {p_head_data}")
                if p_head_data and len(p_head_data) > 0:
                    # logger.info(f"第 {layer_idx} 层端头发现 {len(p_head_data)} 根物料！")
                    base_coord = p_head_data[0]
                    logger.info(f"第 {layer_idx} 层端头锁定目标物料坐标: {base_coord}")

                    # # 将相机相对坐标转换为法兰基座标系绝对坐标
                    # for raw_coord in p_head_data:
                    #     # align_camera=True 表示让相机镜头对准端头，而不是夹爪
                    #     base_coord = self.transform_tool_coord(raw_coord, align_camera=1)
                    #     if base_coord:
                    #         valid_materials_base_coords.append(base_coord)

                    target_material_base_coord = base_coord
                    found_layer_idx = layer_idx
                    # 保存层数记忆 (因为每次都重新拍整层，所以只记层数即可)
                    self.loading_index.current_head_layer_index = layer_idx
                    self.loading_index.save_search_index(const.head_layer_index_name, layer_idx)
                    break  # 跳出 while 重试循环
                else:
                    logger.error("视觉返回OK，但无法获取端头坐标数据")
                    # 数据异常，视为 NG 处理
                    vision_res = "ERROR"

            # ==========================================
            # 分支 2：空视野 (EMPTY) -> 处理垫木或空车
            # ==========================================
            if vision_res == "EMPTY":
                # 检查这一层是否是刚刚被我们抓空的
                if layer_idx == self.loading_index.last_picked_layer:
                    logger.info(f"第 {layer_idx} 层刚刚被抓空，露出垫木！")

                    # 判断当前层是否是料架的 "最底层"
                    is_bottom_layer = (layer_idx == len(head_points) - 1)

                    if is_bottom_layer:
                        # ---------------------------------------------
                        # 情景 A: 最底层抓空 -> 整车全空
                        # ---------------------------------------------
                        logger.critical(f"最底层 (第 {layer_idx} 层) 已被抓空，整车物料全部抓完！")

                        # 移动到最高处安全点 (通常是第0层端头点，或读取 empty_points)
                        empty_pts_cfg = self.cfg_manager.get_process_config(hex(process_addr).upper()).get(
                            "empty_points", [])
                        safe_pt = empty_pts_cfg[0] if empty_pts_cfg else head_points[0]

                        if not self._move_segment_to_target(process_addr, target_point=safe_pt):
                            return False

                        # 发送 21 (空料车报警)
                        self.plc.write_register(process_addr, 21)
                        logger.info(f"发送空料车报警 (21)，已经撤离至安全点...")

                        logger.info("已到达安全点，阻塞死等人工推入新料车并按下复位(20)...")
                        if self.wait_for_plc_val(process_addr, 20, timeout=-1):
                            logger.info("收到复位信号 20，准备重新开始找料")
                            # 重置所有记忆
                            self.loading_index.current_head_layer_index = 0
                            self.loading_index.reset_search_index(const.head_layer_index_name, 0)
                            # 重置抓取记录
                            self.loading_index.last_picked_layer = -1
                            self.loading_index.reset_search_index(const.last_picked_layer_name, -1)

                            # 回退 layer_idx，重新从最高层开始扫描
                            layer_idx = 0
                            continue  # 直接进入下一轮 while 循环
                        else:
                            return False  # 急停打断
                    else:
                        # # 挂起死等人工取垫木
                        # if not self._handle_wood_stick_removal(process_addr, layer_idx):
                        #     return False  # 急停打断
                        #
                        # # 人工取走垫木，按复位恢复后，清除标志位，避免重复报警
                        # # self.last_picked_layer = -1
                        # self.loading_index.last_picked_layer = -1
                        # self.loading_index.reset_search_index(const.last_picked_layer_name, -1)

                        # ---------------------------------------------
                        # 情景 B: 普通层抓空 -> 提示取垫木
                        # ---------------------------------------------
                        logger.info(f"挂起死等人工取走第 {layer_idx} 层垫木...")

                        # 触发 UI 提示 (如果在别处有绑定信号的话)
                        self.sig_wood_stick_alarm.emit(layer_idx)

                        # 发送 22 (取垫木报警)
                        self.plc.write_register(process_addr, 22)

                        logger.info("阻塞死等人工取垫木并按下复位(20)...")
                        if self.wait_for_plc_val(process_addr, 20, timeout=-1):
                            logger.info("收到复位信号 20，垫木已移除")
                            # 清除上次抓取标志
                            self.loading_index.last_picked_layer = -1
                            self.loading_index.reset_search_index(const.last_picked_layer_name, -1)

                            # 层数加1，进入下一层
                            layer_idx += 1
                            continue  # 进入下一轮 while 循环
                        else:
                            return False  # 急停打断

                        logger.info(f"第 {layer_idx} 层垫木已移除，准备下探到新楼层...")
                else:
                    # 只是路过空层 (比如刚开机前两层就是空的)
                    # 不是刚刚抓空的层，只是路过空层 (比如开机时上两层是空的)
                    logger.info(f"第 {layer_idx} 层视野无料 (路过跳过)，准备下探...")
                    layer_idx += 1
                    continue

            # ==========================================
            # 分支 3：真异常 (ERROR)
            # ==========================================
            if vision_res == "ERROR":
                self.plc.write_register(process_addr, 16)
                logger.warning("端头视觉异常 (16)，等待复位 (20)...")
                if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                    continue
                else:
                    return False

        # =======================================================
        # 异常退出保护：如果上面的 while 循环走完了但没找到料
        # =======================================================
        if found_layer_idx == -1:
            # 走到这里通常是因为直接从配置或者记忆读取了一个错误的超限索引，或者7层全空跳过了。
            # 为了防止死锁，直接发空车报警，并重置回0
            logger.critical("所有层遍历完毕仍未找到端头，强制触发空料车报警！")

            self.plc.write_register(process_addr, 21)
            empty_pts_cfg = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("empty_points", [])
            safe_pt = empty_pts_cfg[0] if empty_pts_cfg else head_points[0]
            self._move_segment_to_target(process_addr, target_point=safe_pt)

            if self.wait_for_plc_val(process_addr, 20, timeout=-1):
                self.loading_index.current_head_layer_index = 0
                self.loading_index.reset_search_index(const.head_layer_index_name)

                # 料架全空换新车时，重置抓取记录
                self.loading_index.last_picked_layer = -1
                self.loading_index.reset_search_index(const.last_picked_layer_name, -1)

                # 由于这已经是最外层的保护了，不方便直接重新开始循环，
                # 返回 False 让外层 loop_once 重新拉起是最安全的。
                return False

            return False

        # =======================================================
        # 阶段 2：直接依据视觉 X 生成精拍点
        # =======================================================
        logger.info("=========================================")
        logger.info("端头已找到！依据视觉 X 坐标平移生成精拍点")

        # # 排序：根据你的抓取策略，决定先抓哪一根。
        # # 假设基座标 X 轴对应物料的横向排列，我们按 X 坐标从小到大排序 (从一侧抓到另一侧)
        # valid_materials_base_coords.sort(key=lambda item: item[0])
        #
        # # 始终取排序后的第一根物料进行抓取
        # target_mat_coord = valid_materials_base_coords[0]
        # mat_x, mat_y, mat_z, mat_r = target_mat_coord
        mat_x, mat_y, mat_z, mat_r = target_material_base_coord

        # X加上平移量，消除视觉中的边缘深度干扰
        # Y 加上平移量
        fine_target_x = mat_x + const.depth_interference_x_offset
        fine_target_y = mat_y + const.product_y_offset  # 例如 + 1020.0

        # 使用 found_layer_idx 获取安全的端头点对象，消除变量未定义警告
        target_head_pt = head_points[found_layer_idx]

        # Z轴高度保持端头拍照点的安全高度，防止平移过程撞机
        safe_z = target_head_pt["coords"][2]
        config_curr = target_head_pt.get("config", "elbow_up")

        # 逆解计算 J4 补偿角，保证姿态不变
        j4 = ScaraKinematics().calculate_motor_r_from_world_angle(
            fine_target_x, fine_target_y, safe_z, const.fine_photo_world_angle,
            self.l1, self.l2, self.z0, self.nn3,
            elbow_config=config_curr
        )

        if j4 is None:
            logger.error(f"计算精拍点 J4 失败，目标可能超出机械臂极限")
            return False

        fine_photo_pt = {
            "name": f"Fine_Photo_L{found_layer_idx}_X{fine_target_x:.0f}",
            "coords": [fine_target_x, fine_target_y, safe_z, j4],
            "config": config_curr,
            "photo": 1  # 触发精拍
        }

        logger.info(f"生成精拍点: X={fine_target_x:.1f}, Y={fine_target_y:.1f}, Z={safe_z:.1f}")

        # =======================================================
        # 阶段 3：执行精拍与抓取
        # =======================================================
        while self.running:
            if self.check_estop(): return False

            # 1. 平移到精拍点
            logger.info(f"移动到精拍点: X={fine_target_x:.1f}, Y={fine_target_y:.1f}, Z={safe_z:.1f}")
            if not self._move_segment_to_target(process_addr, target_point=fine_photo_pt):
                return False

            # 2. 在精拍点触发真实的单根物料抓取识别 (算出夹爪的真实抓取坐标)
            final_vision_res = self.handle_vision_recursive_v1(process_addr, fine_photo_pt, loading,
                                                               photo_type=const.photo_type_loading)

            if final_vision_res == "OK":
                logger.info("精确定位成功！最终抓取坐标已保存。")
                self.last_motion_end_point = fine_photo_pt

                # 记录本次成功抓取的层数！
                self.loading_index.last_picked_layer = found_layer_idx
                self.loading_index.reset_search_index(const.last_picked_layer_name, found_layer_idx)

                self.plc.write_register(process_addr, 13)
                return True

            elif final_vision_res == "EMPTY":
                logger.error("精拍发现位置为空！(可能因视觉 X 畸变导致跑偏)")
                self.plc.write_register(process_addr, 16)
                if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                    # return False  # 退出，重新拍端头找
                    continue  # 可能是光线引起深度丢失，还是要重新拍照
                return False


            else:
                self.plc.write_register(process_addr, 16)
                logger.warning("精拍异常 (16)，等待复位 (20)...")
                if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                    continue
                return False

        return False

    # 设备初始化
    @process_action(action_name="设备初始化", action_message="设备初始化",
                    process_step=const.process_step_remove_debris)
    def handle_process_0x400A7(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # plc_addr = self.plc.map_modbus_address(process_addr)
        # logger.info(f"动作{hex(process_addr)} - {plc_addr} 收到请求 {value}，开始执行流程: 设备初始化")

        # 整体逻辑，从当前位置，回到所有动作的初始原点
        # 1 获取当前坐标
        process_start_point = self.get_realtime_point()

        # 2 获取并标准化 Origin 点
        origin_cfg = self.cfg_manager.get_origin_params()
        # 确保 origin 是个标准的点结构
        # origin_point = {
        #     "name": origin_cfg.get("name", "Origin"),
        #     "coords": origin_cfg.get("coords", [0, 0, 0, 0]),
        #     "photo": 0
        # }
        # origin_point = origin_cfg

        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40082/262274
        last_process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        # 取最后一个点的坐标，作为当前流程的
        origin_point = last_process_points[-1]

        # 3 构建点位列表
        points = [process_start_point, origin_point]
        logger.info(f"handle_process_0x400A7, points: {points}")

        # 4 循环发送坐标
        self.execute_standard_motion_sequence(process_addr, points)

        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 首次初始位去缓存位拍照
    @process_action(action_name="首次初始位去缓存位拍照", action_message="首次初始位去缓存位拍照",
                    process_step=const.process_step_detection)
    def handle_process_0x40082(self, process_addr, value):
        # 状态码为整数

        # if value != 10:
        #     return

        # plc_addr = self.plc.map_modbus_address(process_addr)
        # logger.info(f"动作{hex(process_addr)} - {plc_addr} 收到请求 {value}，开始执行流程: 首次初始位去缓存位拍照")

        # 起始点使用实时点位
        process_start_point = self.get_realtime_point()
        # if not process_start_point:
        #     # 如果没有，使用原点
        #     # 1. 获取并标准化 Origin 点
        #     origin_cfg = self.cfg_manager.get_origin_params()
        #     # 确保 origin 是个标准的点结构
        #     origin_point = {
        #         "name": origin_cfg.get("name", "Origin"),
        #         "coords": origin_cfg.get("coords", [0, 0, 0, 0]),
        #         "photo": 0
        #     }
        #     process_start_point = origin_point

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)
        logger.info(f"point list: {points}")

        self.execute_standard_motion_sequence(process_addr, points, photo_type=const.photo_type_normal)

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # # 上料拍照位回门前等待位
    # def handle_process_0x40083(self, process_addr, value):
    #     if value != 10:
    #         return
    #
    #     logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:上料拍照位回门前等待位")
    #
    #     # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
    #     last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40082/262274
    #     last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points", [])
    #     # 取最后一个点的坐标，作为当前流程的
    #     process_start_point = last_process_points[-1]
    #
    #     # 2. 构建点位列表 [Origin, P1, P2...]
    #     points = [process_start_point]
    #     logger.info(f"process address: {process_addr} : {process_start_point['name']}")
    #     process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
    #     if not points:
    #         logger.error("未找到点位配置")
    #         return
    #
    #     points.extend(process_points)
    #     logger.info(f"point list: {points}")
    #
    #     # 执行运动控制
    #     self.execute_standard_motion_sequence(process_addr, points)
    #
    #     # 动作全部成功完成后，更新全局记录
    #     # 将当前动作的最后一个点，标记为下一次动作的起点
    #     self.last_motion_end_point = points[-1]
    #     logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 工装气缸夹具拍照
    @process_action(action_name="工装气缸夹具拍照", action_message="工装放料位识别工装夹爪",
                    process_step=const.process_step_detection)
    def handle_process_0x40084(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # plc_addr = self.plc.map_modbus_address(process_addr)
        # logger.info(f"动作{hex(process_addr)} - {plc_addr} 收到请求 {value}，开始执行流程: 首次臂去下料位拍照")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40083/262275
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的
        # process_start_point = last_process_points[-1]
        process_start_point = self.get_realtime_point()

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)
        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points, photo_type=const.photo_type_cylinder,
                                              vision_retry=True)

        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 缓存拍照位回初始位
    @process_action(action_name="缓存拍照位回初始位", action_message="缓存拍照位回初始位",
                    process_step=const.process_step_remove_debris)
    def handle_process_0x40085(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # plc_addr = self.plc.map_modbus_address(process_addr)
        # logger.info(f"动作{hex(process_addr)} - {plc_addr} 收到请求 {value}，开始执行流程: 缓存拍照位回初始位")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40084/262276
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的
        # process_start_point = last_process_points[-1]
        process_start_point = self.get_realtime_point()

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)
        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points)

        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 初始位去工装位吹气（加工完开门）
    @process_action(action_name="初始位去工装位吹气（加工完开门）", action_message="初始位去工装位吹气（等待加工完开门）",
                    process_step=const.process_step_remove_debris)
    def handle_process_0x40086(self, process_addr, value):
        # if value != 10:
        #     return
        # plc_addr = self.plc.map_modbus_address(process_addr)
        # logger.info(f"动作{hex(process_addr)} - {plc_addr} 收到请求 {value}，开始执行流程: 初始位去工装位吹气（加工完开门）")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40085/262277
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # # 取最后一个点的坐标，作为当前流程的
        # process_start_point = last_process_points[-1]

        process_start_point = self.get_realtime_point()

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)
        points_count = len(points)

        logger.info(f"point list: {points}")
        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points)

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 工装吹气位去工装位拍照
    @process_action(action_name="工装吹气位去工装位拍照", action_message="工装吹气位去工装位拍照",
                    process_step=const.process_step_detection)
    def handle_process_0x40087(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去加工料位拍照")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40086/262278
        last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的
        # process_start_point = last_process_points[-1]

        process_start_point = self.get_realtime_point()
        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)

        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points, photo_type=const.photo_type_normal)

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    #  工装拍照位去工装位取料
    @process_action(action_name="工装拍照位去工装位取料", action_message="工装拍照位去工装位取料",
                    process_step=const.process_step_blanking)
    def handle_process_0x40088(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去加工位取料")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40087/262279
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的
        # process_start_point = last_process_points[-1]
        real_point = self.get_realtime_point()

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [real_point]
        logger.info(f"process address: {process_addr} : {real_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)

        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points)

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 工装取料位去缓存位
    @process_action(action_name="工装取料位去缓存位放料", action_message="工装取料位去缓存位放料",
                    process_step=const.process_step_blanking)
    def handle_process_0x40089(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去缓存位")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40088/262280
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # # 取最后一个点的坐标，作为当前流程的
        # process_start_point = last_process_points[-1]

        real_point = self.get_realtime_point()

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [real_point]
        logger.info(f"process address: {process_addr} : {real_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)
        # points_count = len(points)

        logger.info(f"point list: {points}")

        # l1, l2, z0, nn3, xe, ye, ze, te, j1_curr, j2_curr, distance, config_curr
        # last_point = points[-1]
        # distance = -50
        #
        # forward_point = self.move_forward(last_point, distance)
        # logger.info(f"foward point: {forward_point}")
        # points.append(forward_point)

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points)

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 缓存位去工装位吹气
    @process_action(action_name="缓存放料位去空装位吹气", action_message="缓存放料位去空装位吹气",
                    process_step=const.process_step_remove_debris)
    def handle_process_0x4008A(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去加工位吹气")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40089/262281
        last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的
        process_start_point = last_process_points[-1]

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)
        points_count = len(points)

        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points)

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 空装吹气位去空装位拍照(铝屑识别)
    @process_action(action_name="空装吹气位去空装位拍照(识别铝屑)", action_message="空装吹气位去空装位拍照(识别铝屑)",
                    process_step=const.process_step_remove_debris)
    def handle_process_0x4008B(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去加工位拍照")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008A/262282
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的
        # process_start_point = last_process_points[-1]

        process_start_point = self.get_realtime_point()

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)
        points_count = len(points)

        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points, photo_type=const.photo_type_aluminum)

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 工装拍照位去上料位拍照
    @process_action(action_name="空装拍照位去上料位拍照", action_message="空装拍照位去上料位拍照",
                    process_step=const.process_step_loading)
    def handle_process_0x4008C(self, process_addr, value):
        """
        上料位置拍照，先从工装位置运动到普通点(安全点)，再执行拍照阵列搜索运动
        :param process_addr:
        :param value:
        :return:
        """

        # if value != 10:
        #     return
        #
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去上料位拍照")

        # =========================================================
        # 1. 优先移动到固定的安全过渡点 P (例如门前等待位)
        # =========================================================

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008B/262283
        # logger.info(f"last process addr is : {last_process_addr}, hex is : {hex(last_process_addr)}")
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")

        # 取实时坐标，作为当前流程的起始坐标
        process_start_point = self.get_realtime_point()

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not process_points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)
        # 执行运动控制
        logger.info(f"---> 阶段 1: 优先执行工装位置 -> 安全固定点: {process_points} <---")
        ps_success = self.execute_standard_motion_sequence(
            process_addr,
            points,
            loading=1,
            photo_type=const.photo_type_loading,
            send_done=False  # 执行完成之后不发送13
        )

        if not ps_success:
            logger.error(f"动作 {hex(process_addr)} 阶段 1 (安全过渡) 失败，终止流程。")
            return

        # # =========================================================
        # # 2. 接着执行阵列搜寻逻辑
        # # =========================================================
        # search_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("search_points", [])
        #
        # if not search_points:
        #     logger.error(f"未找到动作 {hex(process_addr)} 的阵列搜寻点位配置")
        #     return
        #
        # logger.info(f"---> 阶段 2: 开始阵列搜寻 (共 {len(search_points)} 个候选点位) <---")
        #
        # # 执行专用的阵列搜寻运动引擎
        # success = self.execute_search_motion_sequence(
        #     process_addr=process_addr,
        #     search_points=search_points,
        #     loading=1,
        #     photo_type=const.photo_type_find_head
        # )
        logger.info(f"---> 阶段 2: 开始执行二阶段定位，端头定位+精拍定位 <---")
        success = self.execute_two_stage_vision_sequence_option_1(
            process_addr=process_addr,
            loading=1
        )

        # 3. 结果处理
        if success:
            logger.info(f"动作 {hex(process_addr)} 搜寻成功！已保存目标物料坐标。")
            # 这里的 self.last_motion_end_point 在引擎内部已经更新过了
        else:
            logger.error(f"动作 {hex(process_addr)} 搜寻终止（缺料、急停或硬件故障）。")

    # 上料拍照位去上料位取料
    @process_action(action_name="上料拍照位去上料位取料", action_message="上料拍照位去上料位取料",
                    process_step=const.process_step_loading)
    def handle_process_0x4008D(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # plc_addr = self.plc.map_modbus_address(process_addr)
        # logger.info(f"动作{hex(process_addr)} - {plc_addr} 收到请求 {value}，开始执行流程:上料拍照位去上料位取料")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008C/262284
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # # 取最后一个点的坐标，作为当前流程的起始坐标
        # process_start_point = last_process_points[-1]

        process_start_point = self.get_realtime_point()

        # 2. 获取上个动作中，视觉给出的源数据地址
        source_addr = last_process_addr

        # 3. 读取视觉数据，eg:[p1, p2, p3]
        vision_head_coords = self.get_vision_data(source_addr, photo_type=const.photo_type_find_head)
        if not vision_head_coords:
            logger.error(f"未找到{hex(source_addr)}的端头视觉数据")

        # vision_points_coords = self.get_vision_data(source_addr, photo_type=const.photo_type_loading)
        vision_loading_coords = self.get_vision_data(source_addr, photo_type=const.photo_type_loading)

        if not vision_loading_coords:
            logger.error(f"未找到地址 {hex(source_addr)} 的抓料视觉数据")
            return

        head_x, head_y, head_z, head_r = vision_head_coords[0]  # 料头坐标
        line_x, line_y, line_z, line_r = vision_loading_coords[0]  # 拍摄的抓取坐标

        # 将视觉坐标转换为标准的点对象格式
        # 构建路径: Start(P0) -> P1 -> P2 -> P3
        # target_points_list = []
        # for idx, coords in enumerate(vision_points_coords):
        #     pt = {
        #         "name": f"Vision_P{idx + 1}",
        #         "coords": coords,
        #         "photo": 0,
        #         "config": process_start_point["config"]
        #     }
        #     target_points_list.append(pt)
        #
        # 目标点
        # target_point = target_points_list[-1]
        logger.info(f"vision head : {vision_head_coords[0]}")
        logger.info(f"vision loading : {vision_loading_coords[0]}")

        # 计算r角偏差之后的真实Y偏差
        world_angle = ScaraKinematics().calculate_world_angle_from_j4(
            line_x, line_y, line_z, line_r, self.l1, self.l2, self.z0, self.nn3, config_type="elbow_up")
        logger.info(f"world angle : {world_angle}")
        world_rad = math.radians(world_angle)
        offset_y = const.product_y_offset * math.cos(world_rad)

        # =======================================================
        # 计算空间位置比例 (Ratio)
        # 纯粹依靠物理位置计算，与视觉漂移彻底解耦
        # =======================================================
        x_ratio = (line_x - const.loading_x_back) / (const.loading_x_front - const.loading_x_back)
        x_ratio = max(0.0, min(x_ratio, 1.0))  # 限制在 0~1 之间

        if x_ratio >= 0.85:
            x_ratio = 1

        # =======================================================
        # X 轴补偿计算 (解决相机侧偏引起的透视+下垂误差)
        # =======================================================
        x_comp = const.loading_x_comp_back + (const.loading_x_comp_front - const.loading_x_comp_back) * x_ratio
        logger.info(f"4008D x_ratio : {x_ratio}, x_comp : {x_comp}")

        # =======================================================
        # Y 轴补偿计算 (解决偏航问题)
        # =======================================================
        # y_comp = const.loading_y_comp_front * x_ratio
        y_comp = const.loading_y_comp_back + (const.loading_y_comp_front - const.loading_y_comp_back) * x_ratio

        # 最终的 Y 坐标 = 视觉真实端头 Y + 1020平移 + 动态补偿
        logger.info(f"4008D offset_y : {offset_y}, y_comp : {y_comp}")

        # 线性插值算出当前槽位需要的角度补偿量
        r_comp = const.loading_r_comp_back + (const.loading_r_comp_front - const.loading_r_comp_back) * x_ratio
        logger.info(f"4008D r_ratio : {x_ratio}, r_comp : {r_comp}")

        # 不补偿
        target_x = line_x
        target_y = head_y + offset_y
        # target_r = line_r

        target_z = line_z

        # 线性补偿
        # target_x = line_x + x_comp
        # target_y = head_y + offset_y + y_comp
        target_r = line_r + r_comp

        target_point = {
            "name": f"Vision_Gripper_P0",
            "coords": [target_x, target_y, target_z, target_r],
            "photo": 0,
            "config": process_start_point["config"]
        }

        logger.info(f"gripper target point is: {target_point}")

        # 构造wp1，目标正上方的点wp1，x,y,r和目标点相同，z和realtime相同
        wp1 = copy.deepcopy(target_point)
        wp1["coords"][2] = process_start_point["coords"][2]

        # wp1向前移动35，构造wp2
        wp2 = self.move_forward(wp1, const.loading_forward_distance)

        # wp2下降到目标点的z, 构造wp3
        h_delta = target_point["coords"][2] - wp1["coords"][2] + 40
        logger.info(f"h_delta is : {h_delta}")

        wp1_1 = self.move_up_down(wp1, -350)

        wp3 = self.move_up_down(wp2, h_delta)

        # wp3下降40，构造wp4
        wp4 = self.move_up_down(wp3, -70)

        # 上方50，测试用
        # way_point_up = self.move_up_down(target_point, 80)
        #
        # way_point_forward = self.move_forward(way_point_up, 45)
        #
        # way_point_down = self.move_up_down(way_point_forward, -100)

        # points = [process_start_point, way_point_up, way_point_forward, way_point_down]
        points = [process_start_point, wp2, wp3, wp4]

        logger.info(f">>>>>>>>>>>>>> point list is : {points}")

        # 执行运动
        self.execute_standard_motion_sequence(process_addr, points)

        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 上料取料去位加工位放料
    @process_action(action_name="上料取料去位空装位放料", action_message="上料取料去位空装位放料",
                    process_step=const.process_step_loading)
    def handle_process_0x4008E(self, process_addr, value):
        # if value != 10:
        #     return
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程: 臂去加工位放料 (含安全回撤)")

        # ==========================================
        # 步骤 1: 确定关键点
        # ==========================================

        # 1.1 起点 (Current): 抓取位 P3
        # 理论上是 self.last_motion_end_point。
        # 为了绝对安全，强烈建议这里读取一次实时坐标！防止PLC夹紧过程中机械臂有微动。
        start_point = self.get_realtime_point()

        # 安全点，抬升到安全高度
        safe_point = copy.deepcopy(start_point)
        safe_point["coords"][2] = const.loading_safe_z

        # 1.3 终点序列 (Target): 放料位配置
        # 读取动作 78 (0x4008E) 自己的配置
        process_points_cfg = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points")
        if not process_points_cfg:
            logger.error("未找到放料点位配置")
            return

        # ==========================================
        # 步骤 2: 拼接完整路径
        # 路径: P3(起点) -> P0(安全点) -> 放料过程点...
        # ==========================================

        # full_sequence = [start_point, p0_point] + process_points_cfg
        full_sequence = [start_point, safe_point] + process_points_cfg

        logger.info(f"生成放料路径: 起点 -> 安全回撤P0 -> 放料点({len(process_points_cfg)}个)")

        # ==========================================
        # 步骤 3: 执行运动
        # ==========================================
        if not self.execute_standard_motion_sequence(process_addr, full_sequence):
            logger.error("放料运动失败")
            return

        # 4. 更新末端记录
        self.last_motion_end_point = full_sequence[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 空装放料位去缓存位检测（关门加工）
    @process_action(action_name="空装放料位去缓存位检测（关门加工）", action_message="空装放料位去缓存位检测（关门加工）",
                    process_step=const.process_step_detection)
    def handle_process_0x4008F(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去缓存拍照位检测")

        # # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008E/262286
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # # 取最后一个点的坐标，作为当前流程的起始坐标
        # process_start_point = last_process_points[-1]

        real_point = self.get_realtime_point()

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [real_point]
        logger.info(f"process address: {process_addr} : {real_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return
        if not process_points:
            logger.error("未找到当前点位配置")
            return

        points.extend(process_points)

        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points)

        # udp_client = UdpClient(
        #     local_port=const.inspection_udp_local_port,
        #     remote_ip=const.inspection_udp_ip,
        #     remote_port=const.inspection_udp_port
        # )
        # udp_client.send_msg("laser_0100_-309.77_537.21_-182.88")
        # udp_client.send_msg("laser_0101_-311.31_599.91_-182.88")
        # udp_client.send_msg("laser_finished")

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 缓存检测去放料位拍照
    @process_action(action_name="缓存检测位去放料位拍照", action_message="缓存检测位去放料位拍照",
                    process_step=const.process_step_palletizing)
    def handle_process_0x40090(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去放料位拍照")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40090/262288
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的起始坐标
        # process_start_point = last_process_points[-1]

        # 取实时点做起始点
        process_start_point = self.get_realtime_point()

        points = [process_start_point]
        # 构建点位列表 [Origin, P1, P2...]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")

        # =======================================================
        # 1. 提取所有过渡点和最终的拍照点
        # =======================================================
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)

        logger.info(f"point list: {points}")

        # 最后一个点就是高空拍照点
        unloading_photo_pt = points[-1]

        # 临时屏蔽这条路径上的所有拍照触发标志, 防止底层引擎擅自调用通用视觉
        for pt in points:
            pt["photo"] = 0

        # =======================================================
        # 2. 安全移动到高空拍照点
        # =======================================================
        logger.info(f"开始前往下料全局拍照点，执行 {len(points) - 1} 段避障轨迹...")
        success = self.execute_standard_motion_sequence(
            process_addr,
            points,
            photo_type=const.photo_type_unloading,
            send_done=False
        )

        if not success:
            logger.error("前往下料拍照点失败或被急停中断，流程终止！")
            return

        # =======================================================
        # 3. 【专属业务状态机】：死循环拍照判定，直到拿到合法位置
        # =======================================================
        target_layer = -1
        target_index = -1
        target_coords = []

        # 防死循环标志位：记录在本次动作中，已经放过垫木的层号
        wood_stick_placed_layer = -1

        while self.running:
            if self.check_estop(): return

            logger.info("已到达拍照位，执行下料区全局拍照识别...")

            # 手动调用 take_photo_position
            vision_res = self.take_photo_position(
                unloading_photo_pt.get("coords"),
                config=unloading_photo_pt.get("config", "elbow_down"),
                ptype=const.photo_type_unloading
            )

            res_status = vision_res.get("res")
            coords = vision_res.get("coords", [])
            layer = vision_res.get("layer", -1)
            index = vision_res.get("position", -1)

            # --- 异常 1: 无下料架 ---
            if layer == 10000:
                self.plc.write_register(process_addr, 21)
                logger.warning("未检测到下料架 (发21)，等待人工推入并复位(20)...")
                if self.wait_for_plc_val(process_addr, 20, timeout=-1):
                    logger.info("收到复位20，重新拍照判定...")
                    continue
                return

            # --- 异常 2: 满料 ---
            if layer == 10001:
                self.plc.write_register(process_addr, 23)
                logger.warning("下料架已满 (发23)，等待人工拉走换车并复位(20)...")
                if self.wait_for_plc_val(process_addr, 20, timeout=-1):
                    logger.info("收到复位20，重新拍照判定...")
                    continue
                return

            # --- 异常 3: 需要放垫木 --- 排除物理层0层
            physical_layer = (const.product_total_layers - 1) - layer

            if index == 0 and physical_layer != 0 and layer != wood_stick_placed_layer:
                self.plc.write_register(process_addr, 22)
                logger.warning(
                    f"新的一层(视觉层号:{layer}, 物理层号:{physical_layer})，需要放置垫木 (发22)，等待复位(20)...")
                if self.wait_for_plc_val(process_addr, 20, timeout=-1):
                    logger.info("收到复位20，重新拍照判定...")
                    # 记录这一层已经放过垫木了，下一轮循环就不会再进这个 if
                    wood_stick_placed_layer = layer
                    continue
                return

            # --- 真异常: 相机断线或算法崩溃 ---
            if res_status != "ok" or layer < 0 or index < 0:
                self.plc.write_register(process_addr, 16)
                logger.error("视觉识别失败或返回数据异常 (发16)，等待复位(20)...")
                if self.wait_for_plc_val(process_addr, 20, timeout=7200):
                    continue
                return

            target_layer = layer
            target_index = index
            target_coords = coords
            logger.info(f"视觉判定成功，分配空位：第 {target_layer} 层，第 {target_index} 列")

            break
            # ============================================

        # 保存数据，供放料逻辑调用
        self.save_vision_data(process_addr, target_coords, photo_type=const.photo_type_unloading,
                              layer=target_layer,
                              index=target_index)

        self.plc.write_register(process_addr, 13)
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 臂缓存位取料
    @process_action(action_name="放料拍照位去缓存位取料", action_message="放料拍照位去缓存位取料",
                    process_step=const.process_step_palletizing)
    def handle_process_0x40091(self, process_addr, value):
        # if value != 10:
        #     return
        #
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去缓存位取料")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008F/262287
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的起始坐标
        # process_start_point = last_process_points[-1]
        process_start_point = self.get_realtime_point()

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)
        points_count = len(points)

        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points)

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 臂去放料位放料
    @process_action(action_name="缓存取料位去放料位放料", action_message="缓存取料位去放料位放料",
                    process_step=const.process_step_palletizing)
    def handle_process_0x40092(self, process_addr, value):
        """
        下料点位的计算，两种方案
        方案一：使用视觉给出的z, 结合X坐标查表生成坐标
        方案二：使用视觉给出的层号layer和索引index，结合下料参数，使用查表法生成坐标
        :param process_addr:
        :param value:
        :return:
        """
        # if value != 10:
        #     return

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40091/262289
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的起始坐标
        # process_start_point = last_process_points[-1]

        process_start_point = self.get_realtime_point()
        points = [process_start_point]

        # 从缓存位走到安全位置
        # 获取点位配置
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not process_points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)

        last_process_point = process_points[-1]
        safe_z = last_process_point["coords"][2]

        # 2. 获取下料拍照动作中，视觉给出的源数据地址
        source_addr = last_process_addr

        # 3. 读取视觉数据，eg:[p1, p2, p3]
        vision_points_data = self.get_vision_data_full(source_addr, const.photo_type_unloading)
        logger.info(f"vision points data for 放料: {vision_points_data}")
        coords = vision_points_data["coords"]
        layer = vision_points_data["layer"]
        index = vision_points_data["position"]
        if not coords:
            logger.error(f"未找到地址 {hex(source_addr)} 的视觉数据")
            return

        #############################################################
        # 使用查表法构造下料坐标
        #############################################################

        # 构造下料点位
        target_x = const.unloding_x_list[index]  # 提取X坐标
        if const.unloding_x_sort == const.unloding_x_sort_desc:
            reverse_x_list = const.unloding_x_list[::-1]
            target_x = reverse_x_list[index]

        target_y = const.unloding_y  # 使用固定Y坐标

        # 提取 Z 坐标 (基于底层高度 + 修正后的132mm层高 * 层数)
        # 注意：这里如果是从底层往上码垛，那就是 layer_0_z + layer * 132

        # 设定一个安全抛料距离，不硬压

        physical_layer = (const.product_total_layers - 1) - layer

        target_z = const.unloading_layer_0_z + (physical_layer * const.unloading_layer_gap)

        # SAFE_DROP_Z = 3.0
        # target_z = const.unloading_layer_0_z + (layer * const.unloading_layer_gap) + SAFE_DROP_Z

        # Z 也可以使用视觉给出的坐标
        # target_z = coords[0][2]

        logger.info(f"查表计算放料目标: 第{physical_layer}层-第{index}列")
        logger.info(f"目标绝对坐标: X={target_x}, Y={target_y}, Z={target_z} (层高132mm)")

        # 计算 J4 角度以维持放料姿态绝对平行
        config_curr = "elbow_down"  # 下料一般用单一姿态

        # 从前往后放，第一个索引位置的elbow需要是up状态
        if const.unloding_x_sort == const.unloding_x_sort_desc and index == 0:
            config_curr = "elbow_up"

        j4 = ScaraKinematics().calculate_motor_r_from_world_angle(
            target_x, target_y, target_z, const.fine_unloading_world_angle,
            self.l1, self.l2, self.z0, self.nn3,
            elbow_config=config_curr
        )
        if not j4:
            logger.info(f"关节4逆解失败")
            return

        final_safe_point = {
            "name": f"Final_safe_point",
            "coords": [target_x, target_y, safe_z, j4],
            "config": config_curr,
            "photo": 0
        }
        points.append(final_safe_point)

        final_unload_target = {
            "name": f"Unload_L{layer}_I{index}",
            "coords": [target_x, target_y, target_z, j4],
            "config": config_curr,
            "photo": 0
        }
        logger.info(f"final unload point: {final_unload_target}")

        points.append(final_unload_target)

        # 执行运动
        motion_ok = self.execute_standard_motion_sequence(process_addr, points)

        # 放料动作成功完成 = 一件产品下线, 才计入当日产量并刷新节拍
        # (execute_standard_motion_sequence 在急停/移动失败/视觉NG时会返回 False, 此时不计产量)
        if motion_ok:
            count, cycle_time = self.kpi_counter.record_one()
            self.sig_kpi.emit(count, cycle_time)
            logger.info(f"[KPI] 放料完成，今日产量={count}，当前节拍={cycle_time:.1f}s")
        else:
            logger.warning("放料运动未成功完成，本次不计入产量")

        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 放料位去初始位
    @process_action(action_name="放料位去初始位", action_message="放料位去初始位",
                    process_step=const.process_step_remove_debris)
    def handle_process_0x40093(self, process_addr, value):
        # if value != 10:
        #     return
        # logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程: 放料位回等待位")

        # ==========================================
        # 步骤 1: 确定关键点
        # ==========================================

        start_point = self.get_realtime_point()

        safe_point = copy.deepcopy(start_point)
        safe_point["coords"][2] = const.unloading_safe_z

        # 1.3 终点序列 (Target): 放料位配置
        # 读取动作 (0x40093) 自己的配置
        process_points_cfg = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points")
        if not process_points_cfg:
            logger.error("未找到放料点位配置")
            return

        # ==========================================
        # 步骤 2: 拼接完整路径
        # 路径: P3(起点) -> P0(安全点) -> 回等待位程点...
        # ==========================================

        full_sequence = [start_point, safe_point] + process_points_cfg
        # full_sequence = [start_point, safe_point]

        logger.info(f"生成放料路径: 起点 -> 安全回撤点 -> 初始位)")

        # ==========================================
        # 步骤 3: 执行运动
        # ==========================================
        if not self.execute_standard_motion_sequence(process_addr, full_sequence):
            logger.error("放料运动失败")
            return

        # 4. 更新末端记录
        self.last_motion_end_point = full_sequence[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

        # ==========================================
        # 5: 周期结束，主动执行相机预防性重启
        # ==========================================
        # if self.vision_service:
        #     # 直接调用专用的硬件重启接口，不影响后台线程
        #     self.vision_service.reboot_camera()

    # # 逻辑判断，10：请求判断  11：物料未加工完成,还有剩余物料  12：加工结束，所有的物料都加工结束；13：取垫木；14：取垫木结束
    # def handle_process_0x40094(self, process_addr, value):
    #     pass
    #
    # def handle_process_0x40095(self, process_addr, value):
    #     #
    #     pass
    #
    # # 等待位去取垫木位拍照
    # def handle_process_0x40096(self, process_addr, value):
    #     #
    #     if value != 10:
    #         return
    #
    #     logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:等待位去取垫木位拍照")
    #
    #     # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
    #     last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40093/262291
    #     last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
    #     # 取最后一个点的坐标，作为当前流程的起始坐标
    #     process_start_point = last_process_points[-1]
    #
    #     # 2. 构建点位列表 [Origin, P1, P2...]
    #     points = [process_start_point]
    #     logger.info(f"process address: {process_addr} : {process_start_point['name']}")
    #     process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
    #     if not points:
    #         logger.error("未找到点位配置")
    #         return
    #
    #     points.extend(process_points)
    #     points_count = len(points)
    #
    #     logger.info(f"point list: {points}")
    #
    #     # 执行运动控制
    #     self.execute_standard_motion_sequence(process_addr, points)
    #
    #     # ============================================
    #     # 动作全部成功完成后，更新全局记录
    #     # 将当前动作的最后一个点，标记为下一次动作的起点
    #     self.last_motion_end_point = points[-1]
    #     logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")
    #
    # # 臂去取垫木
    # def handle_process_0x40097(self, process_addr, value):
    #     if value != 10:
    #         return
    #
    #     logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去取垫木")
    #
    #     # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
    #     # 上一个动作为0x40096
    #     last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40096/262294
    #     # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr)).get("points")
    #     # 取最后一个点的坐标，作为当前流程的起始坐标
    #     # process_start_point = last_process_points[-1]
    #
    #     process_start_point = self.get_realtime_point()
    #
    #     # 2. 获取上个动作中，视觉给出的源数据地址
    #     source_addr = last_process_addr
    #
    #     # 3. 读取视觉数据，eg:[p1, p2, p3]
    #     vision_points_coords = self.get_vision_data(source_addr)
    #     if not vision_points_coords:
    #         logger.error(f"未找到地址 {hex(source_addr)} 的视觉数据")
    #         return
    #
    #     # 将视觉坐标转换为标准的点对象格式
    #     # 构建路径: Start(P0) -> P1 -> P2 -> P3
    #     target_points_list = []
    #     for idx, coords in enumerate(vision_points_coords):
    #         pt = {
    #             "name": f"Vision_P{idx + 1}",
    #             "coords": coords,
    #             "photo": 0
    #         }
    #         target_points_list.append(pt)
    #
    #         # relative_point = self.process_camera_result_to_plc_data(coords)
    #         # relative_point["name"] = f"Vision_P{idx + 1}"
    #         # target_points_list.append(relative_point)
    #
    #     points = [process_start_point] + target_points_list
    #
    #     # 执行运动
    #     self.execute_standard_motion_sequence(process_addr, points)
    #
    #     self.last_motion_end_point = points[-1]
    #     logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")
    #
    # # 取垫木位去放垫木位拍照
    # def handle_process_0x40098(self, process_addr, value):
    #     if value != 10:
    #         return
    #
    #     logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程: 取垫木位去放垫木位拍照")
    #
    #     # ==========================================
    #     # 步骤 1: 确定关键点
    #     # ==========================================
    #
    #     # 1.1 起点 (Current): 抓取位 P3
    #     # 理论上是 self.last_motion_end_point。
    #     # 为了绝对安全，强烈建议这里读取一次实时坐标！防止PLC夹紧过程中机械臂有微动。放料之后，返回中间点PO的过程中，可能要加偏移坐标
    #     start_point = self.get_realtime_point()
    #
    #     ###############################################
    #     # 以start_point为基准base点，添加偏移offset坐标
    #     ###############################################
    #
    #     # 1.2 安全中间点 (Safe): P0
    #     # P0 是动作 (0x40096) 的最后一个点
    #     addr_p0 = 0x40096
    #     cfg_p0 = self.cfg_manager.get_process_config(hex(addr_p0).upper()).get("points")
    #     if not cfg_p0:
    #         logger.error("无法获取安全回撤点(动作76配置为空)")
    #         return
    #
    #     # 构造 P0 点对象
    #     p0_coords = cfg_p0[-1]["coords"]
    #     p0_point = {
    #         "name": "Safe_Retract_P0",
    #         "coords": p0_coords,
    #         "photo": 0
    #     }
    #
    #     # 1.3 终点序列 (Target): 放料位配置
    #     # 读取动作 (0x40098) 自己的配置
    #     process_points_cfg = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points")
    #     if not process_points_cfg:
    #         logger.error("未找到放料点位配置")
    #         return
    #
    #     # ==========================================
    #     # 步骤 2: 拼接完整路径
    #     # 路径: P3(起点) -> P0(安全点) -> 回等待位程点...
    #     # ==========================================
    #
    #     full_sequence = [start_point, p0_point] + process_points_cfg
    #
    #     logger.info(f"生成放料路径: 起点 -> 安全回撤P0 -> 放料点({len(process_points_cfg)}个)")
    #
    #     # ==========================================
    #     # 步骤 3: 执行运动
    #     # ==========================================
    #     if not self.execute_standard_motion_sequence(process_addr, full_sequence):
    #         logger.error("放料运动失败")
    #         return
    #
    #     # 4. 更新末端记录
    #     self.last_motion_end_point = full_sequence[-1]
    #     logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")
    #
    # # 臂去放垫木
    # def handle_process_0x40099(self, process_addr, value):
    #     if value != 10:
    #         return
    #
    #     logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程: 臂去放垫木")
    #
    #     # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
    #     # 上一个动作为0x40098
    #     last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40098/262296
    #     # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr)).get("points")
    #     # 取最后一个点的坐标，作为当前流程的起始坐标
    #     # process_start_point = last_process_points[-1]
    #
    #     process_start_point = self.get_realtime_point()
    #
    #     # 2. 获取上个动作中，视觉给出的源数据地址
    #     source_addr = last_process_addr
    #
    #     # 3. 读取视觉数据，eg:[p1, p2, p3]
    #     vision_points_coords = self.get_vision_data(source_addr)
    #     if not vision_points_coords:
    #         logger.error(f"未找到地址 {hex(source_addr)} 的视觉数据")
    #         return
    #
    #     # 将视觉坐标转换为标准的点对象格式
    #     # 构建路径: Start(P0) -> P1 -> P2 -> P3
    #     target_points_list = []
    #     for idx, coords in enumerate(vision_points_coords):
    #         pt = {
    #             "name": f"Vision_P{idx + 1}",
    #             "coords": coords,
    #             "photo": 0
    #         }
    #         target_points_list.append(pt)
    #
    #         # relative_point = self.process_camera_result_to_plc_data(coords)
    #         # relative_point["name"] = f"Vision_P{idx + 1}"
    #         # target_points_list.append(relative_point)
    #
    #     points = [process_start_point] + target_points_list
    #
    #     # 执行运动
    #     self.execute_standard_motion_sequence(process_addr, points)
    #
    #     self.last_motion_end_point = points[-1]
    #     logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")
    #
    # # 放垫木位去等待位
    # def handle_process_0x4009A(self, process_addr, value):
    #     if value != 10:
    #         return
    #     if value:
    #         return
    #
    #     logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程: 放垫木位去等待位")
    #
    #     # ==========================================
    #     # 步骤 1: 确定关键点
    #     # ==========================================
    #
    #     # 1.1 起点 (Current): 抓取位 P3
    #     # 理论上是 self.last_motion_end_point。
    #     # 为了绝对安全，强烈建议这里读取一次实时坐标！防止PLC夹紧过程中机械臂有微动。放料之后，返回中间点PO的过程中，可能要加偏移坐标
    #     start_point = self.get_realtime_point()
    #
    #     ###############################################
    #     # 以start_point为基准base点，添加偏移offset坐标
    #     ###############################################
    #
    #     # 1.2 安全中间点 (Safe): P0
    #     # P0 是动作 (0x40098) 的最后一个点
    #     addr_p0 = 0x40098
    #     cfg_p0 = self.cfg_manager.get_process_config(hex(addr_p0).upper()).get("points")
    #     if not cfg_p0:
    #         logger.error("无法获取安全回撤点(动作76配置为空)")
    #         return
    #
    #     # 构造 P0 点对象
    #     p0_coords = cfg_p0[-1]["coords"]
    #     p0_point = {
    #         "name": "Safe_Retract_P0",
    #         "coords": p0_coords,
    #         "photo": 0
    #     }
    #
    #     # 1.3 终点序列 (Target): 放料位配置
    #     # 读取动作 (0x4009A) 自己的配置
    #     process_points_cfg = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points")
    #     if not process_points_cfg:
    #         logger.error("未找到放料点位配置")
    #         return
    #
    #     # ==========================================
    #     # 步骤 2: 拼接完整路径
    #     # 路径: P3(起点) -> P0(安全点) -> 回等待位程点...
    #     # ==========================================
    #
    #     full_sequence = [start_point, p0_point] + process_points_cfg
    #
    #     logger.info(f"生成放料路径: 起点 -> 安全回撤P0 -> 放料点({len(process_points_cfg)}个)")
    #
    #     # ==========================================
    #     # 步骤 3: 执行运动
    #     # ==========================================
    #     if not self.execute_standard_motion_sequence(process_addr, full_sequence):
    #         logger.error("放料运动失败")
    #         return
    #
    #     # 4. 更新末端记录
    #     self.last_motion_end_point = full_sequence[-1]
    #     logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")
    #
    # def handle_process_0x4009C(self, process_addr, value):
    #     if value != 10:
    #         return
    #
    #     logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂从缓存拍照位去等待位置")
    #
    #     # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
    #     last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008F/262287
    #     last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
    #     # 取最后一个点的坐标，作为当前流程的
    #     process_start_point = last_process_points[-1]
    #
    #     # 2. 构建点位列表 [Origin, P1, P2...]
    #     points = [process_start_point]
    #     logger.info(f"process address: {process_addr} : {process_start_point['name']}")
    #     process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
    #     if not points:
    #         logger.error("未找到点位配置")
    #         return
    #
    #     points.extend(process_points)
    #     points_count = len(points)
    #
    #     logger.info(f"point list: {points}")
    #     # 执行运动控制
    #     self.execute_standard_motion_sequence(process_addr, points)
    #
    #     # ============================================
    #     # 动作全部成功完成后，更新全局记录
    #     # 将当前动作的最后一个点，标记为下一次动作的起点
    #     self.last_motion_end_point = points[-1]
    #     logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")


def main():
    camera_coord = [-192.65, 32.65, 400, 0.0]
    # camera_coord = [28.08, 30.89, 344, 0.29]
    head_pt = {
        "name": "L1_Head",
        "coords": [
            -53.41,
            259.91,
            70.07,
            -115.54
        ],
        "photo": 0,
        "config": "elbow_up"
    }

    control_obj = Controller()

    control_obj.robot_params = control_obj.cfg_manager.get_robot_params()
    control_obj.l1 = control_obj.robot_params.get('l1')
    control_obj.l2 = control_obj.robot_params.get('l2')
    control_obj.z0 = control_obj.robot_params.get('z0')
    control_obj.nn3 = control_obj.robot_params.get('nn3')

    # plc_cfg = control_obj.cfg_manager.get_plc_config()
    # control_obj.plc = PLCClient(plc_cfg["ip"], plc_cfg["port"])
    #
    # logger.info("机器人后台控制服务启动...")
    # if not control_obj.plc.connect():
    #     logger.error("PLC 连接失败，线程退出")
    #     return  # 连接失败直接退出线程

    target_mat_coord = control_obj.transform_tool_coord(camera_coord, align_camera=1, joint_valid=False)
    print(f"target_mat_coord is: {target_mat_coord}")
    mat_x, mat_y, mat_z, mat_r = target_mat_coord
    y_offset_dist = const.product_y_offset

    # 【核心逻辑】：X 直接用视觉结果，Y 加上平移量
    fine_target_x = mat_x
    fine_target_y = mat_y + y_offset_dist  # 例如 + 1020.0

    # Z轴高度保持端头拍照点的安全高度，防止平移过程撞机
    safe_z = head_pt["coords"][2]
    config_curr = head_pt.get("config", "elbow_up")

    # 逆解计算 J4 补偿角，保证姿态不变
    j4 = ScaraKinematics().calculate_motor_r_from_world_angle(
        fine_target_x, fine_target_y, safe_z, const.fine_photo_world_angle,
        control_obj.l1, control_obj.l2, control_obj.z0, control_obj.nn3,
        elbow_config=config_curr
    )

    if j4 is None:
        logger.error(f"计算精拍点 J4 失败，目标可能超出机械臂极限")
        return False

    found_layer_idx = 0
    fine_photo_pt = {
        "name": f"Fine_Photo_L{found_layer_idx}_X{fine_target_x:.0f}",
        "coords": [fine_target_x, fine_target_y, safe_z, j4],
        "config": config_curr,
        "photo": 1  # 触发精拍
    }

    logger.info(f"生成精拍点: X={fine_target_x:.1f}, Y={fine_target_y:.1f}, Z={safe_z:.1f}")
    print(f"精拍点: {fine_photo_pt}")
    logger.info(fine_photo_pt)

    xe, ye, ze, te = fine_photo_pt["coords"]
    l1 = control_obj.l1
    l2 = control_obj.l2
    z0 = control_obj.z0
    nn3 = control_obj.nn3

    ik = ScaraKinematics().inverse_kinematics_v2(xe, ye, ze, te, l1, l2, z0, nn3, config_type='elbow_up')
    if not ik:
        print(f"目标点逆解失败")
    else:
        print(f"目标点可达: {ik}")

    control_obj.load_vision_file()
    co = control_obj.get_vision_data_full(0x4008c, 2)
    print(co)

    control_obj.handle_process_0x400A7("0x400A7", 10)


if __name__ == "__main__":
    main()
