import sys
import traceback
import time
import threading
import cv2
import gc
import numpy as np
from pyorbbecsdk import (AlignFilter, Pipeline, Config, Context, OBError,
                         OBFormat, OBStreamType, OBAlignMode, OBSensorType)
from src.utils import logger

class OrbbecCameraDevice:
    def __init__(self, width=1280, height=800, fps=10):
        # 不再使用 set_device_changed_callback，坚决不用底层 C++ 回调，极易引发崩溃
        
        self.ctx = Context()
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.config = None
        self.align_filter = None
        # 状态标志
        self.is_connected = False

        # 增加递归锁，防止 keep_alive 线程和业务线程冲突
        self._lock = threading.RLock()

        # 重连冷却时间控制
        self._last_reconnect_time = 0
        self._reconnect_cooldown = 3.0  # 3秒内禁止重复发起连接

    # def _on_device_changed(self, removed_list, added_list):
    #     """设备插拔回调（底层SDK通知）"""
    #     for dev in added_list:
    #         logger.info(f"[Camera Hotplug] Device added: {dev.get_name()}")
    #     for dev in removed_list:
    #         logger.warning(f"[Camera Hotplug] Device removed: {dev.get_name()}")
    #         self.is_connected = False  # 设备拔出，标记配置失效

    def _init_hardware_resources(self):
        """内部函数：创建Pipeline和Config对象，并配置流 (调用前必须加锁)"""
        try:
            # 再次检查设备数量
            if self.ctx.query_devices().get_count() == 0:
                return False, "No device found"

            # 【关键】只有确定有设备了，才创建 Pipeline
            if self.pipeline is None:
                self.pipeline = Pipeline()

            if self.config is None:
                self.config = Config()

            if self.align_filter is None:
                self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)



            # 1. 获取彩色传感器(COLOR_SENSOR)的配置列表
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if not color_profiles:
                return False, "No Color Sensor"

            color_profile = None

            # formats_to_try = [OBFormat.MJPG, OBFormat.RGB]
            formats_to_try = [OBFormat.RGB, OBFormat.YUYV, OBFormat.MJPG]
            # 尝试队列：MJPG -> RGB -> 默认, 1280*720不支持RGB, 优先使用MJPG
            for fmt in formats_to_try:
                try:
                    color_profile = color_profiles.get_video_stream_profile(self.width, self.height, fmt, self.fps)
                    # color_profile = color_profiles.get_video_stream_profile(
                    #     640, 480, fmt, self.fps
                    # )
                    if color_profile:
                        logger.info(f"Matched Color profile: {fmt.name} {self.width}x{self.height} @{self.fps}fps")
                        break
                except:
                    continue

            if color_profile:
                self.config.enable_stream(color_profile)
            else:
                logger.warning("No exact Color profile match! Using default COLOR_STREAM.")
                self.config.enable_stream(OBStreamType.COLOR_STREAM)

            # 2. 获取深度传感器(DEPTH_SENSOR)的配置列表
            try:
                depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                # 深度图通常使用 Y16 格式
                depth_profile = depth_profiles.get_video_stream_profile(self.width, self.height, OBFormat.Y16, self.fps)
                # depth_profile = depth_profiles.get_video_stream_profile(
                #     848, 480, OBFormat.Y16, self.fps
                # )
                self.config.enable_stream(depth_profile)
                # logger.info(f"Depth stream enabled: {self.width}x{self.height} @{self.fps}fps")
            except Exception as e:
                logger.warning(f"Warning: Specific Depth profile not supported ({e}), using default.")
                self.config.enable_stream(OBStreamType.DEPTH_STREAM)

            logger.info("=== Configuration Complete ===")

            return True, "Ready"
        except OBError as e:
            return False, f"SDK Error: {e}"
        except Exception as e:
            return False, f"Setup Error: {e}"

    # def _setup_streams(self):
    #     """修复后的配置函数：使用 OBSensorType"""
    #     try:
    #         # 先检查是否有设备，没设备直接跳过配置
    #         if self.ctx.query_devices().get_count() == 0:
    #             logger.warning("No Orbbec device found during setup_streams.")
    #             return False
    #
    #         # 1. 获取彩色传感器(COLOR_SENSOR)的配置列表
    #         color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    #         color_profile = None
    #
    #         # formats_to_try = [OBFormat.MJPG, OBFormat.RGB]
    #         formats_to_try = [OBFormat.RGB, OBFormat.YUYV]
    #         # 尝试队列：MJPG -> RGB -> 默认, 1280*720不支持RGB, 优先使用MJPG
    #         for fmt in formats_to_try:
    #             try:
    #                 color_profile = color_profiles.get_video_stream_profile(
    #                     self.width, self.height, fmt, self.fps
    #                 )
    #                 if color_profile:
    #                     logger.info(f"Matched Color profile: {fmt.name} {self.width}x{self.height} @{self.fps}fps")
    #                     break
    #             except:
    #                 continue
    #
    #         if color_profile:
    #             self.config.enable_stream(color_profile)
    #         else:
    #             logger.warning("No exact Color profile match! Using default COLOR_STREAM.")
    #             self.config.enable_stream(OBStreamType.COLOR_STREAM)
    #
    #
    #         # 2. 获取深度传感器(DEPTH_SENSOR)的配置列表
    #         try:
    #             depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    #             # 深度图通常使用 Y16 格式
    #             depth_profile = depth_profiles.get_video_stream_profile(
    #                 self.width, self.height, OBFormat.Y16, self.fps
    #             )
    #             self.config.enable_stream(depth_profile)
    #             logger.info(f"Depth stream enabled: {self.width}x{self.height} @{self.fps}fps")
    #         except Exception as e:
    #             logger.warning(f"Warning: Specific Depth profile not supported ({e}), using default.")
    #             self.config.enable_stream(OBStreamType.DEPTH_STREAM)
    #
    #         # 3. 设置软件对齐
    #         # gemini 336l 不支持720p下的硬件对齐(OBAlignMode.HW_MODE)
    #         logger.info("Setting alignment mode to SW_MODE...")
    #         # self.config.set_align_mode(OBAlignMode.SW_MODE)
    #
    #         # 标记配置成功
    #         self.is_stream_configured = True
    #         return True
    #
    #     except OBError as e:
    #         logger.error(f"SDK OBError during setup: {e} \n traceback: {traceback.format_exc()}")
    #         return False
    #     except Exception as e:
    #         logger.error(f"SDK Error during setup: {e} \n traceback: {traceback.format_exc()}")
    #         return False

    def connect(self):
        """启动流水线, 支持热插拔(加锁保护)"""
        with self._lock:

            # 【防重连风暴】：判断是否在冷却时间内
            now = time.time()
            if now - self._last_reconnect_time < self._reconnect_cooldown:
                return False, "In reconnect cooldown"
            self._last_reconnect_time = now

            if self.ctx is None:
                self.ctx = Context()

            ######################################
            # 开启固件日志，下面的代码注释掉
            # 在工作环境目录D:\workspace\projects\robot-single-thread\.venv\Lib\site-packages
            # 文件OrbbecSDKConfig.xml中，FileLogLevel改为0即可
            ######################################
            # device_list = self.ctx.query_devices()
            #
            # dev = device_list.get_device_by_index(0)
            # # logger.info(f"Firmware Version: {dev.get_device_info().firmware_version()}")
            # try:
            #     dev.enable_firmware_log(True)
            #     logger.info("Enable firmware log")
            # except Exception as e:
            #     logger.warning(e)

            # 新增：等待设备枚举，最多等 timeout 秒
            deadline = time.time() + 15
            while time.time() < deadline:
                count = self.ctx.query_devices().get_count()
                if count > 0:
                    break
                time.sleep(0.5)

            if count == 0:
                self.disconnect()
                self.ctx = None
                return False, "No device connected"

            try:
                # 如果 Context 丢了，重新创建
                if not hasattr(self, "ctx") or self.ctx is None:
                    self.ctx = Context()

                # 检查是否有设备
                self.ctx.enable_net_device_enumeration(True)
                device_list = self.ctx.query_devices()
                if device_list.get_count() == 0:
                    # self.is_connected = False
                    self.disconnect()

                    # 彻底销毁 Context！如果没插相机，强行销毁上下文。
                    # 这样下次循环再调用 connect 时，SDK 会被逼着从零重新扫描 Windows 的 USB 端口枚举，
                    # 绝对不会受旧缓存的干扰，完美解决“插上了但软件认不出”的灵异问题。
                    self.ctx = None

                    return False, "No device connected"

                # =========================================================
                # 3. 【新增：僵尸设备探测区】
                # 虽然系统看到了设备，但它可能已经处于死机(Bad Magic)状态
                # =========================================================
                try:
                    # 获取真实的物理设备对象
                    dev = device_list.get_device_by_index(0)
                    # 主动向相机要一次 DeviceInfo，强制触发 USB 底层真实通讯！
                    # 如果此时相机内部芯片已死，这里会立刻抛出 OBError
                    _ = dev.get_device_info()
                except Exception as dev_e:
                    logger.critical(f"捕获到假死/僵尸设备 (可能出现 Bad Magic): {dev_e}")
                    self.disconnect()
                    # 既然设备死了，必须把 Context 扬了，迫使下次重建触发 USB 重置
                    self.ctx = None
                    # 强行休息 2 秒，给 Windows 足够的时间把这个废弃的 USB 节点踢下线
                    time.sleep(2.0)
                    return False, "Zombie device detected and killed"
                # =========================================================

                # 如果之前是处于已连接但卡死的状态，必须先执行一次硬清理
                # 不要尝试直接去 stop 一个旧的 pipeline，直接走毁灭流程重建最安全
                if not self.is_connected and self.pipeline is not None:
                    self.disconnect()

                # 如果还没有 Pipeline 或 Config，进行硬件资源初始化
                if self.pipeline is None:
                    success, msg = self._init_hardware_resources()
                    if not success:
                        self.disconnect()
                        return False, msg

                # 启动流
                try:
                    self.pipeline.enable_frame_sync()
                    self.pipeline.start(self.config)
                    self.is_connected = True

                    return True, "Success"
                except OBError as e:
                    # 如果启动失败（例如被占用），清理资源以便下次重试
                    self.disconnect()
                    return False, f"Start OBError: {e}"

            except Exception as e:
                self.disconnect()
                # 终极防线：遇到意料之外的严重崩溃，连 Context 一起销毁，保证环境纯净
                self.ctx = None
                logger.error(f"SDK Error during connect: {e}")
                return False, str(e)

    def _clean_resources(self):
        """ 绝对隔离的资源释放，确保 Python 侧废弃所有引用"""
        self.is_connected = False

        # 物理防崩溃检查
        # 如果物理设备已经拔出，C++ 底层的 USB 句柄已经失效。
        # 此时调用 stop() 会导致 SDK 内部抛出 Bad Magic 甚至直接段错误闪退。
        # device_physically_exists = False
        # try:
        #     if self.ctx.query_devices().get_count() > 0:
        #         device_physically_exists = True
        # except:
        #     pass

        # if self.pipeline is not None:
        #     if device_physically_exists:
        #         try:
        #             # 尝试优雅停止。
        #             # 如果此时 USB 已断开，这里必然抛出 bad magic 或超时的 OBError
        #             self.pipeline.stop()
        #         except OBError as e:
        #             logger.warning(f"底层相机停止异常 (OBError已吞噬): {e}")
        #         except Exception as e:
        #             logger.warning(f"底层相机停止异常 (Exception已吞噬): {e}")
        #         # finally:
        #         #     # 【终极保命符】
        #         #     # 无论 stop() 成功与否，强制销毁 Python 侧的 pipeline 引用。
        #         #     # 这样会触发底层的 C++ 析构函数，强行释放卡死的句柄。
        #         #     self.pipeline = None
        #
        #     # 【终极保命符】
        #     # 无论如何，强制销毁 Python 侧的 pipeline 引用，这会触发底层安全的析构回收，而不是主动发 stop 指令
        #     self.pipeline = None

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                logger.error(f"底层相机停止异常 (Exception已吞噬): {e}")
                pass  # 忽略一切停止异常 (Device response with bad magic 等)
            finally:
                # 无论 stop 成功与否，必须执行：
                self.pipeline = None

        # 强制销毁其他相关配置对象
        self.config = None
        # self.align_filter = None
        # 显式销毁对齐滤波器，释放可能泄漏的 C++ 内存
        if self.align_filter is not None:
            del self.align_filter
            self.align_filter = None

        self.ctx = None

    def disconnect(self):
        """主动断开设备"""
        with self._lock:
            try:
                self._clean_resources()
            except Exception as e:
                # 万一 C++ 析构时发生了严重错误导致抛出异常，在这里被最后一道防线拦住
                logger.error(f"Disconnect 发生严重崩溃并被成功拦截: {e}\n{traceback.format_exc()}")

    # def disconnect(self):
    #     """关闭设备"""
    #     self.is_connected = False
    #     if self.pipeline:
    #         try:
    #             self.pipeline.stop()
    #         except OBError:
    #             pass
    #         self.pipeline = None
    #     self.config = None
    #     self.align_filter = None

    def haraware_reset_simulation(self):
        """模拟重启主板"""
        with self._lock:
            logger.critical("hardware reset start")

            self.disconnect()

            self.ctx = None

            self.is_connected = False

            time.sleep(8)

    def hardware_reset(self):
        """
        【核弹级恢复】当遇到 setXu failed 等底层死锁时，强制重启相机主板
        """
        with self._lock:
            logger.critical(">>> 准备执行相机底层硬件重启 (Hardware Reboot) <<<")
            try:
                if self.ctx is not None:
                    device_list = self.ctx.query_devices()
                    if device_list.get_count() > 0:
                        # 获取物理设备对象
                        dev = device_list.get_device_by_index(0)
                        logger.critical("向相机发送 Reboot 指令...")
                        dev.reboot()  # 调用底层 C++ 接口强行重启相机主板
                        logger.critical("Reboot 指令发送成功，相机即将掉线重建。")
            except Exception as e:
                logger.error(f"硬件重启指令发送失败: {e}")
            finally:
                # 无论指令是否发送成功，彻底清空上位机的内存对象
                self._clean_resources()
                self.ctx = None  # 杀掉 Context，逼迫 SDK 下次重新枚举 USB 树
                self.is_connected = False

    # def hardware_reset(self):
    #
    #     with self._lock:
    #
    #         logger.critical("hardware reboot start")
    #
    #         old_ctx = self.ctx
    #
    #         # 先把Python对象全部废掉
    #         self.pipeline = None
    #         self.config = None
    #         self.align_filter = None
    #
    #         self.ctx = None
    #         self.is_connected = False
    #
    #         gc.collect()
    #
    #         try:
    #             if old_ctx:
    #
    #                 devs = old_ctx.query_devices()
    #
    #                 if devs.get_count() > 0:
    #                     dev = devs.get_device_by_index(0)
    #
    #                     logger.critical("send reboot")
    #
    #                     dev.reboot()
    #
    #                     logger.critical("reboot success")
    #
    #         except Exception as e:
    #             logger.error(e)

    # def hardware_reset(self):
    #
    #     with self._lock:
    #
    #         logger.critical("hardware reboot start")
    #
    #         old_ctx = self.ctx
    #
    #         try:
    #             # 先释放SDK对象
    #             self._clean_resources()
    #
    #             time.sleep(0.5)
    #
    #             if old_ctx:
    #                 devs = old_ctx.query_devices()
    #
    #                 if devs.get_count() > 0:
    #                     dev = devs.get_device_by_index(0)
    #                     logger.critical("向相机发送 Reboot 指令...")
    #                     dev.reboot()
    #                     logger.critical("Reboot 指令发送成功，相机即将掉线重建。")
    #         except Exception as e:
    #             logger.error(e)
    #
    #         finally:
    #             self.ctx = None
    #             self.is_connected = False

    # 底层直接转化为 Numpy 并销毁 C++ 对象
    def _convert_color_frame(self, color_frame):
        f_width = color_frame.get_width()
        f_height = color_frame.get_height()
        color_format = color_frame.get_format()
        raw_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

        if color_format == OBFormat.MJPG:
            # 如果是 MJPG，使用 OpenCV 解码成 BGR
            bgr_img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            if bgr_img is None: return None
            # 转换为 RGB (因为之前的逻辑是存图前转 BGR，或者算法需要 RGB)
            return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        elif color_format == OBFormat.RGB:
            # 只有格式确实是 RGB 时才能直接 reshape
            return raw_data.reshape((f_height, f_width, 3)).copy()
        return None

    def _convert_depth_frame(self, depth_frame):
        d_width = depth_frame.get_width()
        d_height = depth_frame.get_height()
        return np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((d_height, d_width)).copy()

    def get_frames(self, timeout_ms=2000):
        """
        获取同步帧数据并【立即转为 Python 格式】
        返回: (status, color_numpy_array, depth_numpy_array)
        """
        with self._lock:
            if not self.pipeline or not self.is_connected:
                return False, None, None

            try:
                logger.info(
                    f"wait_for_frames begin pipeline={id(self.pipeline)}"
                )
                frames = self.pipeline.wait_for_frames(timeout_ms)
                if not frames:
                    return False, None, None
                logger.info(
                    f"wait_for_frames end frames={frames is not None}"
                )

                # 软对齐可能会因为光线突变偶发失败
                if self.align_filter:
                    try:
                        aligned_frames = self.align_filter.process(frames)
                        if aligned_frames:
                            frames = aligned_frames.as_frame_set()
                        else:
                            # 如果对齐失败，记录警告，但不要断开，直接视为这一次取帧失效
                            #  SDK 对齐失败，直接跳过当前帧，不认定为断线
                            logger.warning("Align filter failed to process frames, skipping this frame.")
                            return False, None, None
                    except OBError as e:
                        logger.warning(f"Align filter OBError: {e}")
                        return False, None, None

                # 对齐处理
                # if self.align_filter:
                #     aligned_frames = self.align_filter.process(frames)
                #     if aligned_frames:
                #         frames = aligned_frames.as_frame_set()

                # 提取彩色和深度帧
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame is not None and depth_frame is not None:
                    # 立即转换为 Python 安全的 Numpy 内存
                    color_img = self._convert_color_frame(color_frame)
                    depth_img = self._convert_depth_frame(depth_frame)

                    # 立即、显式释放 C++ 底层引用！绝不让它活着离开这个函数
                    del color_frame
                    del depth_frame
                    del frames

                    if color_img is not None and depth_img is not None:
                        return True, color_img, depth_img

                return False, None, None

            except Exception as e:
                logger.error(f"Wait for frames error (Device Disconnected?): {e}\n{traceback.format_exc()}")
                # 一旦报错，立刻将自身状态置为断开，下次调用自动走重连
                self.disconnect()
                return False, None, None

    def flush_frames(self, num_frames=5):
        """
        排空历史缓存帧，获取最新画面，并给自动曝光留出时间
        :param num_frames: 丢弃的帧数 (建议 3~5 帧)
        """
        with self._lock:
            if not self.pipeline or not self.is_connected:
                return

            logger.info(f"清理相机底层缓存队列，丢弃前 {num_frames} 帧...")
            for _ in range(num_frames):
                try:
                    # 设定极短的超时时间，把积压在内存里的图迅速抽走抛弃
                    # 在 10fps 下，100ms 的超时极大概率会什么都抽不到！ 必须放宽到 200ms 左右。
                    # FPS=10 时，每帧需要 100ms 物理生成时间。
                    # 超时必须大于 100ms，这里设为 200ms 最安全！
                    frames = self.pipeline.wait_for_frames(200)
                    if frames:
                        # 显式释放内存
                        color = frames.get_color_frame()
                        depth = frames.get_depth_frame()
                        if color: del color
                        if depth: del depth
                        del frames
                except Exception as e:
                    logger.info(f"flush frames error: {e}\n {traceback.format_exc()}")
                    pass
            logger.info(f"flush frames end")

    def is_alive(self):
        """简单的链路健康检查"""
        with self._lock:
            try:
                # 尝试通过获取设备信息来判断链路是否正常
                if self.is_connected and self.pipeline and self.ctx and self.ctx.query_devices().get_count() > 0:
                    return True
            except Exception as e:
                logger.info(f"camera is alive error: {e}\n {traceback.format_exc()}")
                pass
            return False

    def check_device_exist(self):
        with self._lock:
            try:
                if self.ctx is None:
                    return False

                return self.ctx.query_devices().get_count() > 0
            except Exception as e:
                logger.info(f"check_device_exist error: {e}\n {traceback.format_exc()}")
                return False

    def diagnose_alignment_issue(self):
        """
        诊断对齐问题的工具函数 - 安全版本
        """
        if not self.pipeline or not self.is_connected:
            logger.error("Pipeline not started or not connected")
            return

        try:
            logger.info("=== Starting Camera Diagnosis ===")

            # 1. 获取设备信息
            try:
                device = self.pipeline.get_device()
                device_info = device.get_device_info()
                logger.info(f"Device Info: {device_info}")
            except Exception as e:
                logger.error(f"Failed to get device info: {e}")

            # 2. 获取帧数据
            logger.info("Attempting to get frames...")
            frames = self.pipeline.wait_for_frames(3000)

            if not frames:
                logger.error("No frames received!")
                return

            logger.info("Frames object obtained successfully")

            # 3. 安全地获取彩色帧信息
            logger.info("--- Checking Color Frame ---")
            color_frame = None
            try:
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    logger.error("Color frame is None!")
                else:
                    # 安全地获取各种属性
                    try:
                        width = color_frame.get_width()
                        height = color_frame.get_height()
                        logger.info(f"Color frame size: {width}x{height}")
                    except Exception as e:
                        logger.error(f"Could not get color frame dimensions: {e}")

                    try:
                        fmt = color_frame.get_format()
                        logger.info(f"Color frame format: {fmt}")
                    except Exception as e:
                        logger.error(f"Could not get color format: {e}")

                    logger.info("Color frame obtained successfully")

            except Exception as e:
                logger.error(f"Error getting color frame: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

            # 4. 安全地获取深度帧信息
            logger.info("--- Checking Depth Frame ---")
            depth_frame = None
            try:
                depth_frame = frames.get_depth_frame()

                # 不要直接打印depth_frame对象，而是检查它的存在
                if depth_frame is None:
                    logger.error("Depth frame is None!")
                else:
                    logger.info("Depth frame object exists")

                    # 安全地获取深度帧的属性
                    try:
                        width = depth_frame.get_width()
                        height = depth_frame.get_height()
                        logger.info(f"Depth frame size: {width}x{height}")
                    except Exception as e:
                        logger.error(f"Could not get depth frame dimensions: {e}")

                    try:
                        fmt = depth_frame.get_format()
                        logger.info(f"Depth frame format: {fmt}")
                    except Exception as e:
                        logger.error(f"Could not get depth format: {e}")

                    # 检查深度数据
                    try:
                        data = depth_frame.get_data()
                        if data is not None:
                            logger.info(f"Depth data size: {len(data)} bytes")
                        else:
                            logger.warning("Depth data is None")
                    except Exception as e:
                        logger.error(f"Could not get depth data: {e}")

                    logger.info("Depth frame obtained successfully")

            except Exception as e:
                logger.error(f"Error getting depth frame: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

            # 5. 对齐检查
            logger.info("--- Alignment Check ---")
            if color_frame is not None and depth_frame is not None:
                try:
                    color_w = color_frame.get_width()
                    color_h = color_frame.get_height()
                    depth_w = depth_frame.get_width()
                    depth_h = depth_frame.get_height()

                    logger.info(f"Color resolution: {color_w}x{color_h}")
                    logger.info(f"Depth resolution: {depth_w}x{depth_h}")

                    if color_w == depth_w and color_h == depth_h:
                        logger.warning("Same resolution - check for sensor offset misalignment")
                        logger.info("Misalignment might be due to physical sensor offset")
                    else:
                        logger.info(f"Different resolutions - alignment mapping needed")
                        logger.info(f"Alignment should handle {depth_w}x{depth_h} -> {color_w}x{color_h}")

                except Exception as e:
                    logger.error(f"Error comparing frames: {e}")
            else:
                logger.error("Cannot compare - one or both frames are None")
                if color_frame is None:
                    logger.error("Color frame is missing")
                if depth_frame is None:
                    logger.error("Depth frame is missing")

            logger.info("=== Diagnosis Complete ===")

        except Exception as e:
            logger.error(f"Diagnosis error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    camera = OrbbecCameraDevice()
    status, msg = camera.connect()
    if status:
        logger.info(f"Connected: {msg}")

        # 先清空缓存
        camera.flush_frames(5)

        # 运行诊断
        # for i in range(5):
        #     camera.diagnose_alignment_issue()
        #     logger.info("\n\n")
    else:
        logger.error(f"Failed to connect: {msg}")
    sys.exit(1)