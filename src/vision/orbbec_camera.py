import traceback
from pyorbbecsdk import (Pipeline, Config, Context, OBError,
                         OBFormat, OBStreamType, OBAlignMode, OBSensorType)
from src.utils import logger

class OrbbecCameraDevice:
    def __init__(self, width=1280, height=720, fps=15):
        self.ctx = Context()
        # 设置设备插拔回调 (可选，但这里我们主要用轮询方式实现)
        self.ctx.set_device_changed_callback(self._on_device_changed)

        self.width = width
        self.height = height
        self.fps = fps

        # self.pipeline = Pipeline()  # 提前初始化 pipeline 以便获取 profile
        # self.config = Config()

        self.pipeline = None
        self.config = None

        # 流配置 标志位
        # self.is_stream_configured = False

        # 状态标志
        self.is_connected = False

        # 获取设备并配置流
        # self._setup_streams()

    def _on_device_changed(self, removed_list, added_list):
        """设备插拔回调（底层SDK通知）"""
        for dev in added_list:
            logger.info(f"[Camera Hotplug] Device added: {dev.get_name()}")
        for dev in removed_list:
            logger.warning(f"[Camera Hotplug] Device removed: {dev.get_name()}")
            self.is_connected = False  # 设备拔出，标记配置失效

    def _init_hardware_resources(self):
        """内部函数：创建Pipeline和Config对象，并配置流"""
        try:
            # 再次检查设备数量
            if self.ctx.query_devices().get_count() == 0:
                return False, "No device found"

            # 【关键】只有确定有设备了，才创建 Pipeline
            if self.pipeline is None:
                self.pipeline = Pipeline()

            if self.config is None:
                self.config = Config()

            # 1. 获取彩色传感器(COLOR_SENSOR)的配置列表
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if not color_profiles:
                return False, "No Color Sensor"

            color_profile = None

            formats_to_try = [OBFormat.MJPG, OBFormat.RGB]
            # 尝试队列：MJPG -> RGB -> 默认, 1280*720不支持RGB, 优先使用MJPG
            for fmt in formats_to_try:
                try:
                    color_profile = color_profiles.get_video_stream_profile(
                        self.width, self.height, fmt, self.fps
                    )
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
                depth_profile = depth_profiles.get_video_stream_profile(
                    self.width, self.height, OBFormat.Y16, self.fps
                )
                self.config.enable_stream(depth_profile)
                logger.info(f"Depth stream enabled: {self.width}x{self.height} @{self.fps}fps")
            except Exception as e:
                logger.warning(f"Warning: Specific Depth profile not supported ({e}), using default.")
                self.config.enable_stream(OBStreamType.DEPTH_STREAM)

            # 3. 设置软件对齐
            # gemini 336l 不支持720p下的硬件对齐(OBAlignMode.HW_MODE)
            logger.info("Setting alignment mode to SW_MODE...")
            self.config.set_align_mode(OBAlignMode.SW_MODE)

            # 标记配置成功
            # self.is_stream_configured = True
            return True, "Ready"
        except OBError as e:
            return False, f"SDK Error: {e}"
        except Exception as e:
            return False, f"Setup Error: {e}"

    def _setup_streams(self):
        """修复后的配置函数：使用 OBSensorType"""
        try:
            # 先检查是否有设备，没设备直接跳过配置
            if self.ctx.query_devices().get_count() == 0:
                logger.warning("No Orbbec device found during setup_streams.")
                return False

            # 1. 获取彩色传感器(COLOR_SENSOR)的配置列表
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = None

            formats_to_try = [OBFormat.MJPG, OBFormat.RGB]
            # 尝试队列：MJPG -> RGB -> 默认, 1280*720不支持RGB, 优先使用MJPG
            for fmt in formats_to_try:
                try:
                    color_profile = color_profiles.get_video_stream_profile(
                        self.width, self.height, fmt, self.fps
                    )
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
                depth_profile = depth_profiles.get_video_stream_profile(
                    self.width, self.height, OBFormat.Y16, self.fps
                )
                self.config.enable_stream(depth_profile)
                logger.info(f"Depth stream enabled: {self.width}x{self.height} @{self.fps}fps")
            except Exception as e:
                logger.warning(f"Warning: Specific Depth profile not supported ({e}), using default.")
                self.config.enable_stream(OBStreamType.DEPTH_STREAM)

            # 3. 设置软件对齐
            # gemini 336l 不支持720p下的硬件对齐(OBAlignMode.HW_MODE)
            logger.info("Setting alignment mode to SW_MODE...")
            self.config.set_align_mode(OBAlignMode.SW_MODE)

            # 标记配置成功
            self.is_stream_configured = True
            return True

        except OBError as e:
            logger.error(f"SDK OBError during setup: {e} \n traceback: {traceback.format_exc()}")
            return False
        except Exception as e:
            logger.error(f"SDK Error during setup: {e} \n traceback: {traceback.format_exc()}")
            return False

    def connect(self):
        """启动流水线, 支持热插拔"""
        try:
            # 检查是否有设备
            device_list = self.ctx.query_devices()
            if device_list.get_count() == 0:
                self.is_connected = False
                self.disconnect()
                return False, "No device connected"

            # if not self.is_stream_configured:
            #     logger.info("Device detected, configuring streams...")
            #     if not self._setup_streams():
            #         return False, "Failed to configure streams"

            # 如果还没有 Pipeline 或 Config，进行硬件资源初始化
            if self.pipeline is None:
                success, msg = self._init_hardware_resources()
                if not success:
                    self.is_connected = False
                    return False, msg

            # 注意：如果 pipeline 已经开启，先停止
            try:
                self.pipeline.stop()
            except:
                pass

            # # 启动 Pipeline
            # self.pipeline.start(self.config)
            # 3. 启动流
            try:
                self.pipeline.start(self.config)
                self.is_connected = True
                return True, "Success"
            except OBError as e:
                # 如果启动失败（例如被占用），清理资源以便下次重试
                self.disconnect()
                return False, str(e)

        # except OBError as e:
        #     # 如果启动失败（比如USB带宽不足，或者配置失效），重置标志位
        #     self.is_stream_configured = False
        #     logger.error(f"SDK OBError Error during connect: {e}")
        #     return False, str(e)
        except Exception as e:
            # self.is_stream_configured = False
            self.disconnect()
            logger.error(f"SDK Error during connect: {e}")
            return False, str(e)

    def disconnect(self):
        """关闭设备"""
        self.is_connected = False
        if self.pipeline:
            try:
                self.pipeline.stop()
            except OBError:
                pass
            self.pipeline = None
        self.config = None

    def get_frames(self, timeout_ms=5000):
        """
        获取一帧同步数据
        返回: (status, color_frame, depth_frame)
        """
        if not self.pipeline or not self.is_connected:
            return False, None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if not frames:
                return False, None, None

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame and depth_frame:
                return True, color_frame, depth_frame
        except OBError as e:
            logger.error(f"Wait for frames error: {e}")

        return False, None, None

    def is_alive(self):
        """简单的链路健康检查"""
        try:
            # 尝试通过获取设备信息来判断链路是否正常
            if self.is_connected and self.pipeline and self.ctx.query_devices().get_count() > 0:
                return True
        except:
            pass
        return False

