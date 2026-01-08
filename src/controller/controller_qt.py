from src.utils.logger import logger
from src.plc.plc_client import PLCClient
from src.utils.config_manager import ConfigManager, CONFIG_FILE, VISION_DATA_FILE
from src.core.kinematics import ScaraKinematics
from src.core.trajectory import TrajectoryV2
from src.consts import const
from src.vision.jxbpipeline import VisionSystem
from PyQt5.QtCore import QThread, pyqtSignal
import time
import json
import os

class Controller(QThread):
    log_signal = pyqtSignal(str)

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

        self.origin_point = self.cfg_manager.get_origin_params()  # 获取原点配置
        self.last_motion_end_point = self.origin_point  # 定义【全局当前坐标记录】，初始化为原点
        # 用于存储实时电机状态 [J1, J2, J3, J4], 初始化为 0.0，供 UI 读取
        self.last_axis_status = [0.0, 0.0, 0.0, 0.0]
        self.last_joint_status = [0.0, 0.0, 0.0, 0.0]

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

    def run(self):
        logger.info("机器人后台控制服务启动...")
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
        if not self.plc.connect():
            logger.error("PLC 连接失败，线程退出")
            return  # 连接失败直接退出线程

        self.running = True  # 标记运行

        while self.running:
            try:
                self.loop_once()
                # 轮询间隔 2s
                # time.sleep(2)
                self.msleep(1500)  # QThread 推荐用 msleep (毫秒)
            except KeyboardInterrupt:
                logger.info("用户停止程序")
                self.running = False
            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                # time.sleep(10)
                self.msleep(1500)

    # 停止方法
    def stop_service(self):
        logger.info("正在停止控制服务...")
        self.running = False
        self.wait()  # 等待线程安全退出
        logger.info("控制服务已停止")

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

    def loop_once(self):
        realtime_point = self.get_realtime_point()
        logger.info(f"current point: {realtime_point}")

        """执行一次完整的轮询和处理"""

        e_stop_regs = self.plc.read_holding_registers(self.plc.map_modbus_address(const.ADDR_ESTOP_MONITOR), 1)
        logger.info(f"emergency addr: {self.plc.map_modbus_address(const.ADDR_ESTOP_MONITOR)}, stop regs : {e_stop_regs}")

        # === 1. 全局急停拦截 ===
        if self.check_estop():
            # 如果检测到急停：
            # 1. 清除可能存在的缓存状态
            self.last_motion_end_point = None
            # 2. 不读取业务寄存器，直接返回
            # 3. 打印日志提示（为了防止刷屏，可以加个状态位控制打印频率）
            logger.info("系统急停中，等待复位...")
            time.sleep(0.5)  # 降低轮询频率
            return

        # === 2. 正常业务流程 ===
        # 1. 批量读取状态寄存器
        states = self.plc.read_holding_registers(
            self.plc.map_modbus_address(const.process_start_addr),
            const.process_num
        )
        logger.info(f"loop states: {states}")
        if not states:
            return

        # 2. 处理映射字典
        handler_map = {
            0x400A7: self.handle_process_0x400A7,
            0x40082: self.handle_process_0x40082,
            0x40083: self.handle_process_0x40083,
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
            0x40094: self.handle_process_0x40094,
            0x40095: self.handle_process_0x40095,
            0x40096: self.handle_process_0x40096,
            0x40097: self.handle_process_0x40097,
            0x40098: self.handle_process_0x40098,
            0x40099: self.handle_process_0x40099,
            0x4009A: self.handle_process_0x4009A,
            0x4009C: self.handle_process_0x4009C,
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

    def get_realtime_point(self):
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

            logger.info("res: {}".format(regs))

            # 2. 解析浮点数 (注意：汇川通常是 Little Word Endian)
            # 假设你的 self.plc.registers_to_float 已经封装好了 struct 处理
            # regs[0:2] -> J1, regs[2:4] -> J2 ...
            curr_j1 = self.plc.registers_to_float(regs[0:2])
            curr_j2 = self.plc.registers_to_float(regs[2:4])
            curr_j3 = self.plc.registers_to_float(regs[4:6])
            curr_j4 = self.plc.registers_to_float(regs[6:8])

            self.last_joint_status = [curr_j1, curr_j2, curr_j3, curr_j4]

            logger.info(f"PLC反馈关节信息: J1={curr_j1:.2f}, J2={curr_j2:.2f}, J3={curr_j3:.2f}, J4={curr_j4:.2f}")

            # 3. 调用正运动学 (FK) 计算 XYZR
            # 注意：需确保 ScaraKinematics 类中有 forward_kinematics 方法
            fk_res = ScaraKinematics().forward_kinematics(
                curr_j1, curr_j2, curr_j3, curr_j4,
                self.l1, self.l2, self.z0, self.nn3
            )
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
            self.plc.read_holding_registers(0x40000, 1)

            addr_j1, addr_j2, addr_j3, addr_j4, addr_vel, addr_acc = const.point_addresses[i]

            # 每次循环都查一下急停，防止发送一半按急停
            if self.check_estop():
                logger.warning("发送过程中急停，终止发送")
                return False

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
                        self.plc.read_holding_registers(0x40000, 1)

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

    # 发送单个坐标，到地址[0x40003, 0x40005, 0x40007, 0x40009, 0x4000B]， 0x4000B用于占位，暂时没有实际意义
    def send_coords_once(self, process_addr, point):
        success_flag = True
        point_address = const.point_addresses[0]
        addr_j1, addr_j2, addr_j3, addr_j4, velocity, accelerated_velocity = point_address
        # addr_j1, addr_j2, addr_j3, addr_j4 = point_address
        coords = point.get("coords", [0, 0, 0, 0])
        xe, ye, ze, te = coords[0], coords[1], coords[2], coords[3]
        try:
            # 逆解运算
            ik_res = ScaraKinematics.inverse_kinematics_v2(xe, ye, ze, te, self.l1, self.l2, self.z0, self.nn3)
            if ik_res:
                # 写入浮点数 (关节角度/位置)
                self.plc.write_float(addr_j1, ik_res['the1'])
                self.plc.write_float(addr_j2, ik_res['the2'])
                self.plc.write_float(addr_j3, ik_res['the3'])
                self.plc.write_float(addr_j4, ik_res['th4'])

                logger.info(
                    f"坐标({xe},{ye}) -> 关节({ik_res['the1']:.2f}, {ik_res['the2']:.2f})")
            else:
                logger.error(f"逆解失败：动作{process_addr}目标点{coords}不可达")
                success_flag = False
                # 逆解失败也填0，防止意外动作
                self._fill_zero_group(addr_j1, addr_j2, addr_j3, addr_j4)
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
            logger.error("相机参数准备失败：逆解无解")
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

    def take_photo(self):
        # 执行拍照动作，拍照结果成功返回OK，异常返回NG, 返回字符串
        return "OK"

    def take_photo_check(self):
        return "OK"

    def take_photo_position(self, camera_coords, config, loading=None):
        """
        :param camera_coords:传给相机的拍照位置坐标
        :param loading, 上下料参数，1/上料，2/下料
        :return:
        {
            "res": "ok/ng"
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
            logger.info(f"camera_coords >>>>>>>> : {camera_coords}, loading : {loading}")
            camera_prepare_coords = self.prepare_params_for_camera({"coords": camera_coords, "config": config})
            logger.info(f"camera_prepare_coords >>>>>> : {camera_prepare_coords}")
            pos = VisionSystem().run(camera_prepare_coords, loading=loading) # 相机只要x,y,z，不要r参数
            return pos
        except Exception as e:
            logger.info(f"take photo position error: {e}")
            return {"res": "error"}


    def save_vision_data(self, process_addr, coords_list):
        """
        保存视觉坐标数据
        :param process_addr: 动作地址 (int)，如 0x4008C
        :param coords_list: 坐标列表 [[x,y,z,r], ...]
        """
        key = hex(process_addr)  # 转成字符串 "0x4008c" 作为 Key

        # 1. 更新内存
        self.vision_data_cache[key] = coords_list

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

            # 更新当前动作的数据
            current_data[key] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "coords": coords_list
            }

            # 写入文件
            with open(VISION_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, indent=4, ensure_ascii=False)

            logger.info(f"视觉数据已保存至 {VISION_DATA_FILE}, Key={key}, 数量={len(coords_list)}")

        except Exception as e:
            logger.error(f"保存视觉数据失败: {e}")

    def get_vision_data(self, process_addr):
        """
        获取视觉坐标数据
        :param process_addr: 产生数据的动作地址 (int)
        :return: coords_list 或 []
        """
        key = hex(process_addr)

        # 1. 优先从内存读
        if key in self.vision_data_cache:
            return self.vision_data_cache[key]

        # 2. 内存没有，尝试从文件重新加载 (应对程序重启的情况)
        self.load_vision_file()
        if key in self.vision_data_cache:
            return self.vision_data_cache[key]

        return []

    def load_vision_file(self):
        """从文件加载视觉坐标数据到内存"""
        if not os.path.exists(VISION_DATA_FILE):
            return
        try:
            with open(VISION_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 简化结构，只存坐标列表到内存
                for k, v in data.items():
                    self.vision_data_cache[k] = v.get("coords", [])
        except Exception as e:
            logger.error(f"读取视觉文件失败: {e}")

    def get_vision_status(self):
        """
        获取状态，判断物料加工状态，以及垫木取/放状态
        11：物料未加工完成,还有剩余物料  12：加工结束，所有的物料都加工结束；13：取垫木；14：取垫木结束
        """
        return 11

    # 辅助：阻塞监听 PLC 寄存器直到变为指定值 (支持超时，但等待复位时通常超时时间设很长或无限)
    def wait_for_plc_val(self, addr, target_val, timeout=10.0):
        start_time = time.time()
        while self.running:
            # 发送一个无意义的读取，仅仅为了保持 TCP 连接活跃
            self.plc.read_holding_registers(0x40000, 1)

            # 1. 优先检查急停
            if self.check_estop():
                logger.warning("任务因急停中断，停止等待")
                return False  # 返回 False，上层逻辑会感知并退出

            # 如果 timeout < 0，则无限等待
            if timeout > 0 and (time.time() - start_time > timeout):
                logger.error(f"等待PLC地址 {hex(addr)} 变为 {target_val} 超时")
                return False

            # 读取当前值
            regs = self.plc.read_holding_registers(addr, 1)
            if regs and regs[0] == target_val:
                return True

            time.sleep(0.1)  # 避免死循环占用过高CPU
        return False

    def _move_segment_to_target(self, process_addr=None, start_point=None, target_point=None):
        """
        辅助函数：从当前位置移动到目标点 (包含获取起点、插值、发送、握手)
        """
        # 1. 获取实时起点
        if not start_point:
            start_point = self.get_realtime_point()

        # 2. 生成插值路径
        interpolated_path = TrajectoryV2().generate_cartesian_interpolated_path(
            [start_point, target_point],
            num_inserts=const.point_interpolated_num
        )
        points_to_send = interpolated_path[1:]  # 去掉起点

        target_name = target_point.get("name", "Temp_Point")
        logger.info(f"执行移动片段 -> {target_name}")

        # 3. 发送数据
        if not self.send_coords_batch(process_addr, points_to_send, 8):
            # 1. 检查急停
            if self.check_estop():
                return False # 直接退出函数，即清除了当前流程数据

            logger.error("坐标发送失败")
            return False

        if self.check_estop(): return False # 确认是否因急停退出
        # 4. 握手 (11)
        self.plc.write_register(process_addr, 11)

        # 5. 等待到位 (12), timeout=-1表示无线等待
        if not self.wait_for_plc_val(process_addr, 12, timeout=10):
            if self.check_estop(): return False # 再次确认是否因急停退出

            logger.error("等待到位超时")
            return False

        # 6. 更新内存中的位置记录
        self.last_motion_end_point = target_point
        return True

    def handle_vision_recursive(self, process_addr, point, loading=None):
        """
        :param process_addr: 动作地址位
        :param point: 拍照坐标
        :param loading, 上下料参数，1/上料，2/下料

        处理迭代式视觉逻辑
        功能：
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
            logger.info(f"执行视觉检测 (第 {loop_count} 次), 当前物理坐标: {camera_coords}, 当前loading动作: {loading}")

            # 1. 调用相机接口
            result = self.take_photo_position(camera_coords, config, loading=loading)
            logger.info(f"photo result is >>>>>>>>: {result}")

            # 2. 解析结果
            res_status = result.get("res", "ng")
            trigger = result.get("trigger", "")
            coords = result.get("coords", [])

            # === 情况 A: 视觉 NG ===
            if res_status != "ok":
                logger.error(f"视觉返回 NG: {res_status}")
                return False

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
                self.save_vision_data(process_addr, coords)

                return True  # 动作 76 任务完成

            else:
                logger.error(f"未知的 trigger 类型: {trigger}")
                return False

        logger.error("视觉重拍次数过多，强制停止")
        return False

    def execute_standard_motion_sequence(self, process_addr, points_sequence, loading=None):
        """
        标准运动序列执行函数 (修改版)
        :param process_addr:动作地址位
        :param points_sequence: 坐标点位
        :param loading: 上下料标记，1/上料，2/下料
        """
        # 1. 检查急停
        if self.check_estop():
            logger.warning("当前处于急停状态，拒绝执行新任务")
            return False

        points_count = len(points_sequence)

        for i in range(points_count - 1):
            start_point = points_sequence[i]
            end_point = points_sequence[i + 1]

            # 获取 photo 标志 (0 或 1)
            photo_trigger = end_point.get("photo", 0)

            # === while 重试循环 (处理 NG -> 16 -> 20 -> 重试) ===
            while self.running:
                # 1. 检查急停
                if self.check_estop():
                    logger.critical("流程强制终止：急停触发")
                    self.last_motion_end_point = None  # 清除记忆点，强制下次重新获取实时位置
                    return False # 直接退出函数，即清除了当前流程数据


                # 1. 执行移动 (使用提取出的通用函数)
                # 这会处理插值、发送、等待12
                if not self._move_segment_to_target(process_addr=process_addr, start_point=start_point, target_point=end_point):
                    return False  # 移动失败(如急停)，直接退出

                # 2. 拍照逻辑处理
                vision_ok = True

                if photo_trigger == 1:
                    # 调用刚才写的递归视觉逻辑
                    # 它内部会处理：拍照 -> (可能移动 -> 重拍) -> 移动到抓取点
                    if self.handle_vision_recursive(process_addr, end_point, loading):
                        vision_ok = True
                        # 注意：如果 handle_vision_recursive 成功，机械臂已经移动到了 p_r1
                        # 此时 self.last_motion_end_point 已经是 p_r1 了
                    else:
                        vision_ok = False

                # 3. 结果分支 (NG 处理)
                if vision_ok:
                    break  # 成功，退出 while，进入下一个 for (如果有的话)
                else:
                    # 发送 NG 信号
                    self.plc.write_register(process_addr, 16)
                    logger.warning("视觉/流程 NG (16)，等待复位 (20)...")

                    # 等待 PLC 复位信号
                    if self.wait_for_plc_val(process_addr, 20, timeout=-1):
                        logger.info("收到 20，重试当前步骤")
                        # 这里的 continue 会导致重新执行 _move_segment_to_target
                        # 也就是重新走到 end_point，然后重新触发拍照
                        continue
                    else:
                        return False

        # 序列全部完成，发送 13
        self.plc.write_register(process_addr, 13)
        return True

    def execute_standard_motion_sequence_v0(self, process_addr, points_sequence):
        """
        通用的运动控制序列执行函数
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
            points_to_send = interpolated_path[1:] # 去掉起点

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
                if not self.wait_for_plc_val(process_addr, 12, timeout=10):
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
                        self.save_vision_data(process_addr, coords_list)
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
                    if self.wait_for_plc_val(process_addr, 20, timeout=-1):
                        continue
                    else:
                        return False

        # 全部完成
        self.plc.write_register(process_addr, 13)
        logger.info("所有点位执行完毕，发送完成信号 13")
        return True

    # 设备初始化
    def handle_process_0x400A7(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:首次初始位去上料位拍照")

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
        origin_point = origin_cfg
        # 3 构建点位列表
        points = [process_start_point, origin_point]
        logger.info(f"handle_process_0x400A7, points: {points}")

        # 4 循环发送坐标，改用execute_standard_motion_sequence，原先的代码注释掉
        self.execute_standard_motion_sequence(process_addr, points)

        """"
        points_count = len(points)
        for i in range(points_count-1):
            start_point = points[i]
            end_point = points[i+1]

            # 生成9个点：[Start, I1...I7, End]
            # num_inserts=7 意味着中间插入7个，总共生成 2+7=9 个点
            interpolated_path  = TrajectoryV2().generate_cartesian_interpolated_path(
                [start_point, end_point],
                num_inserts=const.point_interpolated_num
            )

            # 切片去掉Start，保留 [I1...I7, End]，共8个点
            points_to_send = interpolated_path[1:]
            logger.info(f"handle_process_0x400A7, points_to_send: {points_to_send}")


            # 获取终点的拍照标志
            photo_trigger = end_point.get("photo", 0)
            target_name = end_point.get("name", f"P{i+1}")

            while self.running:
                # logger.info(f"开始执行第 {i + 1}/{points_count} 个点位: {end_point}")
                logger.info(f"--- 执行段 {i + 1}/{points_count - 1}: {start_point.get('name')} -> {target_name} ---")

                # 1. 发送坐标
                # send_res = self.send_coords_once(process_addr, point)
                send_res = self.send_coords_batch(process_addr, points_to_send, 8)
                if not send_res:
                    logger.error("坐标发送失败，终止流程")
                    return  # 或者根据需求处理

                # 2. 写入 11 (坐标已发送)
                self.plc.write_register(process_addr, 11)
                logger.info(f"坐标发送成功，发送响应11")

                # 3. 等待 PLC 到位 (等待 12)
                # 这里的超时时间是机械臂运动时间，需合理设置
                if not self.wait_for_plc_val(process_addr, 12, timeout=-1):
                    logger.error("等待机械臂到位超时")
                    return
                logger.info(f"段 {target_name} 到位, plc回复 12)")

                break
        # ============================================
        # === 所有点位循环结束 ===/
        logger.info("所有点位执行完毕，发送完成信号 13")
        self.plc.write_register(process_addr, 13)
        """

        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 首次初始位去上料位拍照
    def handle_process_0x40082(self, process_addr, value):
        # 状态码为整数
        # value = raw_values[0], 或者使用下面的参数
        # value = self.plc.registers_to_int32(raw_values)
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:首次初始位去上料位拍照")

        # 起始点定义为上一个动作的结束点
        # process_start_point = self.last_motion_end_point
        process_start_point = self.get_realtime_point()
        if not process_start_point:
            # 如果没有，使用原点
            # 1. 获取并标准化 Origin 点
            origin_cfg = self.cfg_manager.get_origin_params()
            # 确保 origin 是个标准的点结构
            origin_point = {
                "name": origin_cfg.get("name", "Origin"),
                "coords": origin_cfg.get("coords", [0, 0, 0, 0]),
                "photo": 0
            }
            process_start_point = origin_point

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return

        points.extend(process_points)
        logger.info(f"point list: {points}")

        self.execute_standard_motion_sequence(process_addr, points)

        """
        points_count = len(points)

        for i in range(points_count-1): # 循环发送坐标
            start_point = points[i]
            end_point = points[i+1]

            # 生成9个点：[Start, I1...I7, End]
            # num_inserts=7 意味着中间插入7个，总共生成 2+7=9 个点
            interpolated_path  = TrajectoryV2().generate_cartesian_interpolated_path(
                [start_point, end_point],
                num_inserts=const.point_interpolated_num
            )

            # 切片去掉Start，保留 [I1...I7, End]，共8个点
            points_to_send = interpolated_path[1:]

            # # 获取终点的拍照标志
            photo_trigger = end_point.get("photo", 0)
            target_name = end_point.get("name", f"P{i+1}")

            while self.running:
                # logger.info(f"开始执行第 {i + 1}/{points_count} 个点位: {end_point}")
                logger.info(f"--- 执行段 {i + 1}/{points_count - 1}: {start_point.get('name')} -> {target_name} ---")

                # 1. 发送坐标
                # send_res = self.send_coords_once(process_addr, point)
                send_res = self.send_coords_batch(process_addr, points_to_send, 8)
                if not send_res:
                    logger.error("坐标发送失败，终止流程")
                    return  # 或者根据需求处理

                # 2. 写入 11 (坐标已发送)
                self.plc.write_register(process_addr, 11)
                logger.info(f"坐标发送成功，发送响应11")

                # 3. 等待 PLC 到位 (等待 12)
                # 这里的超时时间是机械臂运动时间，需合理设置
                if not self.wait_for_plc_val(process_addr, 12, timeout=-1):
                    logger.error("等待机械臂到位超时")
                    return
                logger.info(f"段 {target_name} 到位, plc回复 12)")

                # 4. 拍照逻辑 (如果有配置)
                vision_ok = True
                if photo_trigger == 1:
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

                # === 5. 结果判断与分支 ===
                if vision_ok:
                    # --- 成功路径 ---
                    # 跳出 while 重试循环，继续 for 循环执行下一个点
                    break
                else:
                    # --- 失败路径 (NG) ---
                    # 1. 发送 16 给 PLC
                    self.plc.write_register(process_addr, 16)
                    logger.warning("已发送 16，等待人工复位 (等待 20)...")

                    # 2. 【关键】死循环等待 20
                    # 这里 timeout 设为 -1 (无限等待)，直到操作员按下复位
                    if self.wait_for_plc_val(process_addr, 20, timeout=-1):
                        logger.info("收到复位信号 20，准备重新执行当前点位")
                        # 3. 收到 20 后，不 break for循环，而是 continue while 循环
                        # 这会导致重新执行 send_coords_once -> write 11 -> wait 12 -> photo
                        continue
                    else:
                        # 如果程序停止运行
                        return


        # === 所有点位循环结束 ===
        logger.info("所有点位执行完毕，发送完成信号 13")
        self.plc.write_register(process_addr, 13)
        """

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 上料拍照位回门前等待位
    def handle_process_0x40083(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:上料拍照位回门前等待位")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40082/262274
        last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points", [])
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
        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points)

        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 首次臂去下料位拍照
    def handle_process_0x40084(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:首次臂去下料位拍照")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40083/262275
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
        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points)

        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 下料拍照位回门前等待位
    def handle_process_0x40085(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:下料拍照位回门前等待位")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40084/262276
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
        logger.info(f"point list: {points}")

        # 执行运动控制
        self.execute_standard_motion_sequence(process_addr, points)

        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 臂去加工料位吹气
    def handle_process_0x40086(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去加工料位吹气")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40085/262277
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

    # 臂去加工料位拍照
    def handle_process_0x40087(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去加工料位拍照")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40086/262278
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

    #  臂去加工位取料
    def handle_process_0x40088(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去加工位取料")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40087/262279
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

    # 臂去缓存位
    def handle_process_0x40089(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去缓存位")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40088/262280
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

    # 臂去加工位吹气
    def handle_process_0x4008A(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去加工位吹气")

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

    # 臂去加工位拍照
    def handle_process_0x4008B(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去加工位拍照")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008A/262282
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

    # 臂去上料位拍照
    def handle_process_0x4008C(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去上料位拍照")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008B/262283
        logger.info(f"last process addr is : {last_process_addr}, hex is : {hex(last_process_addr)}")
        last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
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
        self.execute_standard_motion_sequence(process_addr, points, loading=1)

        # ============================================
        # 动作全部成功完成后，更新全局记录
        # 将当前动作的最后一个点，标记为下一次动作的起点
        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 臂去上料位取料
    def handle_process_0x4008D(self, process_addr, value):
        if value != 10:
            return

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008C/262284
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # # 取最后一个点的坐标，作为当前流程的起始坐标
        # process_start_point = last_process_points[-1]

        process_start_point = self.get_realtime_point()

        # 2. 获取上个动作中，视觉给出的源数据地址
        source_addr = last_process_addr

        # 3. 读取视觉数据，eg:[p1, p2, p3]
        vision_points_coords = self.get_vision_data(source_addr)
        if not vision_points_coords:
            logger.error(f"未找到地址 {hex(source_addr)} 的视觉数据")
            return

        # 将视觉坐标转换为标准的点对象格式
        # 构建路径: Start(P0) -> P1 -> P2 -> P3
        target_points_list = []
        for idx, coords in enumerate(vision_points_coords):
            # pt = {
            #     "name": f"Vision_P{idx + 1}",
            #     "coords": coords,
            #     "photo": 0
            # }
            # target_points_list.append(pt)

            relative_point = self.process_camera_result_to_plc_data(coords)
            relative_point["name"] = f"Vision_P{idx + 1}"
            target_points_list.append(relative_point)

        points = [process_start_point] + target_points_list
        logger.info(f"point list: {points}")

        # 执行运动
        self.execute_standard_motion_sequence(process_addr, points)

        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 臂去加工位放料
    def handle_process_0x4008E(self, process_addr, value):
        if value != 10:
            return
        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程: 臂去加工位放料 (含安全回撤)")

        # ==========================================
        # 步骤 1: 确定关键点
        # ==========================================

        # 1.1 起点 (Current): 抓取位 P3
        # 理论上是 self.last_motion_end_point。
        # 为了绝对安全，强烈建议这里读取一次实时坐标！防止PLC夹紧过程中机械臂有微动。
        start_point = self.get_realtime_point()

        # 1.2 安全中间点 (Safe): P0
        # P0 是动作 76 (0x4008C) 的最后一个点
        addr_p0 = 0x4008C
        cfg_p0 = self.cfg_manager.get_process_config(hex(addr_p0).upper()).get("points")
        if not cfg_p0:
            logger.error("无法获取安全回撤点(动作76配置为空)")
            return

        # 构造 P0 点对象
        p0_coords = cfg_p0[-1]["coords"]
        p0_point = {
            "name": "Safe_Retract_P0",
            "coords": p0_coords,
            "photo": 0
        }

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

        full_sequence = [start_point, p0_point] + process_points_cfg

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

    # 臂去缓存拍照位检测
    def handle_process_0x4008F(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去缓存拍照位检测")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008E/262286
        last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的起始坐标
        process_start_point = last_process_points[-1]

        # 2. 构建点位列表 [Origin, P1, P2...]
        points = [process_start_point]
        logger.info(f"process address: {process_addr} : {process_start_point['name']}")
        process_points = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points", [])
        if not points:
            logger.error("未找到点位配置")
            return
        if not process_points:
            logger.error("未找到当前点位配置")
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

    # 臂缓存位取料
    def handle_process_0x40090(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂缓存位取料")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008F/262287
        last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的起始坐标
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

    # 臂去放料位拍照
    def handle_process_0x40091(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去放料位拍照")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40090/262288
        last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的起始坐标
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

    # 臂去放料位放料
    def handle_process_0x40092(self, process_addr, value):
        if value != 10:
            return

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40091/262289
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的起始坐标
        # process_start_point = last_process_points[-1]

        process_start_point = self.get_realtime_point()

        # 2. 获取上个动作中，视觉给出的源数据地址
        source_addr = last_process_addr

        # 3. 读取视觉数据，eg:[p1, p2, p3]
        vision_points_coords = self.get_vision_data(source_addr)
        if not vision_points_coords:
            logger.error(f"未找到地址 {hex(source_addr)} 的视觉数据")
            return

        # 将视觉坐标转换为标准的点对象格式
        # 构建路径: Start(P0) -> P1 -> P2 -> P3
        target_points_list = []
        for idx, coords in enumerate(vision_points_coords):
            pt = {
                "name": f"Vision_P{idx + 1}",
                "coords": coords,
                "photo": 0
            }
            target_points_list.append(pt)

        points = [process_start_point] + target_points_list

        # 执行运动
        self.execute_standard_motion_sequence(process_addr, points)

        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 放料位回等待位
    def handle_process_0x40093(self, process_addr, value):
        if value != 10:
            return
        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程: 放料位回等待位")

        # ==========================================
        # 步骤 1: 确定关键点
        # ==========================================

        # 1.1 起点 (Current): 抓取位 P3
        # 理论上是 self.last_motion_end_point。
        # 为了绝对安全，强烈建议这里读取一次实时坐标！防止PLC夹紧过程中机械臂有微动。放料之后，返回中间点PO的过程中，可能要加偏移坐标
        start_point = self.get_realtime_point()

        ###############################################
        # 以start_point为基准base点，添加偏移offset坐标
        ###############################################

        # 1.2 安全中间点 (Safe): P0
        # P0 是动作 (0x40091) 的最后一个点
        addr_p0 = 0x40091
        cfg_p0 = self.cfg_manager.get_process_config(hex(addr_p0).upper()).get("points")
        if not cfg_p0:
            logger.error("无法获取安全回撤点(动作76配置为空)")
            return

        # 构造 P0 点对象
        p0_coords = cfg_p0[-1]["coords"]
        p0_point = {
            "name": "Safe_Retract_P0",
            "coords": p0_coords,
            "photo": 0
        }

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

        full_sequence = [start_point, p0_point] + process_points_cfg

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

    # 逻辑判断，10：请求判断  11：物料未加工完成,还有剩余物料  12：加工结束，所有的物料都加工结束；13：取垫木；14：取垫木结束
    def handle_process_0x40094(self, process_addr, value):
        if value != 10:
            return

        status = self.get_vision_status()
        self.plc.write_register(process_addr, status)

    def handle_process_0x40095(self, process_addr, value):
        #
        pass

    # 等待位去取垫木位拍照
    def handle_process_0x40096(self, process_addr, value):
        #
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:等待位去取垫木位拍照")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40093/262291
        last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr).upper()).get("points")
        # 取最后一个点的坐标，作为当前流程的起始坐标
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

    # 臂去取垫木
    def handle_process_0x40097(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂去取垫木")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # 上一个动作为0x40096
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40096/262294
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr)).get("points")
        # 取最后一个点的坐标，作为当前流程的起始坐标
        # process_start_point = last_process_points[-1]

        process_start_point = self.get_realtime_point()

        # 2. 获取上个动作中，视觉给出的源数据地址
        source_addr = last_process_addr

        # 3. 读取视觉数据，eg:[p1, p2, p3]
        vision_points_coords = self.get_vision_data(source_addr)
        if not vision_points_coords:
            logger.error(f"未找到地址 {hex(source_addr)} 的视觉数据")
            return

        # 将视觉坐标转换为标准的点对象格式
        # 构建路径: Start(P0) -> P1 -> P2 -> P3
        target_points_list = []
        for idx, coords in enumerate(vision_points_coords):
            pt = {
                "name": f"Vision_P{idx + 1}",
                "coords": coords,
                "photo": 0
            }
            target_points_list.append(pt)

        points = [process_start_point] + target_points_list

        # 执行运动
        self.execute_standard_motion_sequence(process_addr, points)

        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 取垫木位去放垫木位拍照
    def handle_process_0x40098(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程: 取垫木位去放垫木位拍照")

        # ==========================================
        # 步骤 1: 确定关键点
        # ==========================================

        # 1.1 起点 (Current): 抓取位 P3
        # 理论上是 self.last_motion_end_point。
        # 为了绝对安全，强烈建议这里读取一次实时坐标！防止PLC夹紧过程中机械臂有微动。放料之后，返回中间点PO的过程中，可能要加偏移坐标
        start_point = self.get_realtime_point()

        ###############################################
        # 以start_point为基准base点，添加偏移offset坐标
        ###############################################

        # 1.2 安全中间点 (Safe): P0
        # P0 是动作 (0x40096) 的最后一个点
        addr_p0 = 0x40096
        cfg_p0 = self.cfg_manager.get_process_config(hex(addr_p0).upper()).get("points")
        if not cfg_p0:
            logger.error("无法获取安全回撤点(动作76配置为空)")
            return

        # 构造 P0 点对象
        p0_coords = cfg_p0[-1]["coords"]
        p0_point = {
            "name": "Safe_Retract_P0",
            "coords": p0_coords,
            "photo": 0
        }

        # 1.3 终点序列 (Target): 放料位配置
        # 读取动作 (0x40098) 自己的配置
        process_points_cfg = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points")
        if not process_points_cfg:
            logger.error("未找到放料点位配置")
            return

        # ==========================================
        # 步骤 2: 拼接完整路径
        # 路径: P3(起点) -> P0(安全点) -> 回等待位程点...
        # ==========================================

        full_sequence = [start_point, p0_point] + process_points_cfg

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

    # 臂去放垫木
    def handle_process_0x40099(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程: 臂去放垫木")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        # 上一个动作为0x40098
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x40098/262296
        # last_process_points = self.cfg_manager.get_process_config(hex(last_process_addr)).get("points")
        # 取最后一个点的坐标，作为当前流程的起始坐标
        # process_start_point = last_process_points[-1]

        process_start_point = self.get_realtime_point()

        # 2. 获取上个动作中，视觉给出的源数据地址
        source_addr = last_process_addr

        # 3. 读取视觉数据，eg:[p1, p2, p3]
        vision_points_coords = self.get_vision_data(source_addr)
        if not vision_points_coords:
            logger.error(f"未找到地址 {hex(source_addr)} 的视觉数据")
            return

        # 将视觉坐标转换为标准的点对象格式
        # 构建路径: Start(P0) -> P1 -> P2 -> P3
        target_points_list = []
        for idx, coords in enumerate(vision_points_coords):
            pt = {
                "name": f"Vision_P{idx + 1}",
                "coords": coords,
                "photo": 0
            }
            target_points_list.append(pt)

        points = [process_start_point] + target_points_list

        # 执行运动
        self.execute_standard_motion_sequence(process_addr, points)

        self.last_motion_end_point = points[-1]
        logger.info(f"动作完成，更新当前位置记录为: {process_addr} : {self.last_motion_end_point['name']}")

    # 放垫木位去等待位
    def handle_process_0x4009A(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程: 放垫木位去等待位")

        # ==========================================
        # 步骤 1: 确定关键点
        # ==========================================

        # 1.1 起点 (Current): 抓取位 P3
        # 理论上是 self.last_motion_end_point。
        # 为了绝对安全，强烈建议这里读取一次实时坐标！防止PLC夹紧过程中机械臂有微动。放料之后，返回中间点PO的过程中，可能要加偏移坐标
        start_point = self.get_realtime_point()

        ###############################################
        # 以start_point为基准base点，添加偏移offset坐标
        ###############################################

        # 1.2 安全中间点 (Safe): P0
        # P0 是动作 (0x40098) 的最后一个点
        addr_p0 = 0x40098
        cfg_p0 = self.cfg_manager.get_process_config(hex(addr_p0).upper()).get("points")
        if not cfg_p0:
            logger.error("无法获取安全回撤点(动作76配置为空)")
            return

        # 构造 P0 点对象
        p0_coords = cfg_p0[-1]["coords"]
        p0_point = {
            "name": "Safe_Retract_P0",
            "coords": p0_coords,
            "photo": 0
        }

        # 1.3 终点序列 (Target): 放料位配置
        # 读取动作 (0x4009A) 自己的配置
        process_points_cfg = self.cfg_manager.get_process_config(hex(process_addr).upper()).get("points")
        if not process_points_cfg:
            logger.error("未找到放料点位配置")
            return

        # ==========================================
        # 步骤 2: 拼接完整路径
        # 路径: P3(起点) -> P0(安全点) -> 回等待位程点...
        # ==========================================

        full_sequence = [start_point, p0_point] + process_points_cfg

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


    def handle_process_0x4009C(self, process_addr, value):
        if value != 10:
            return

        logger.info(f"动作{hex(process_addr)} 收到请求 {value}，开始执行流程:臂从缓存拍照位去等待位置")

        # 1. 起始点定义为上一个动作的结束点, 固定上一个动作的plc地址位，暂时不使用全局self.last_motion_end_point
        last_process_addr = const.last_process_addr_map.get(process_addr)  # 0x4008F/262287
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