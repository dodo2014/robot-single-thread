import os
import cv2
import numpy as np
import json
import time
import queue
import threading
import traceback
from pyorbbecsdk import (OBFormat)
from src.vision.orbbec_camera import OrbbecCameraDevice
from src.depthSegmentPython.RGBDDepthSegmenterWrap import RGBDDetector
from src.utils.path_helper import get_camera_img_dir, get_logs_dir
from src.utils import logger
from src.utils import detector_logger
from src.consts import const

# 导入编译好的 C++ 模块 (cpp_algo.so)
try:
    import cpp_algo
except ImportError:
    logger.warning("Warning: cpp_algo module not found.")


class DetectAlgoService:
    def __init__(self, product_no: str, save_dir: str = get_camera_img_dir()):
        self.product_no = product_no
        self.save_dir = save_dir
        self.device = OrbbecCameraDevice()
        self.max_retries = 10
        self.depth_show = 1
        # 最新帧缓存 (供 HMI 界面读取)
        self._last_color_img = None   # numpy.ndarray (H, W, 3) RGB
        self._last_result_img = None  # numpy.ndarray 检测结果叠加图
        # 初始化 C++ 算法
        # self.algo = cpp_algo.MaterialAlgorithm()
        # init_res = self.algo.initialize(self.product_no)
        # if init_res.get("code") != 0:
        #     raise RuntimeError(f"Algorithm Init Failed: {init_res}")

        self.detector = RGBDDetector()
        detector_init_res = self.detector.init(product_no)
        if detector_init_res.get("code") != 0:
            raise RuntimeError(f"Algorithm Init Failed: {detector_init_res}")

        # 异步存图队列与线程
        self.save_jpg = 0
        self.save_queue = queue.Queue(maxsize=100)  # 限制队列长度防止内存溢出
        self.stop_event = threading.Event()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        self.alive_thread = threading.Thread(target=self._keep_alive_worker, daemon=True)
        self.alive_thread.start()

        self._keepalive_fail_count = 0

        self.init_connect()

    def init_connect(self):
        # 初始化相机并预热
        logger.info("Initializing camera hardware...")
        try:
            success, msg = self.device.connect()
            if not success:
                # 启动时没插相机，只警告，不抛异常，不退出
                logger.warning(f"Camera init failed: {msg}. Hot-plug supported - waiting for device..")
            else:
                # 只有连接成功才预热
                self._warm_up()
        except Exception as e:
            logger.error(f"Unexpected error during camera init: {e}")

    def _warm_up(self):
        """抽取出预热逻辑：持续获取有效帧，直到帧数达标，模拟官方示例的 while True + continue 语义"""
        logger.info("Warming up camera...")
        warmed = 0
        target = 5  # 预热自动曝光, 10fps下建议target = 5，原先的20耗时太长
        max_attempts = 100
        attempt = 0
        while warmed < target and attempt < max_attempts:
            try:
                success, color_img, depth_img = self.device.get_frames(timeout_ms=500)
                if success:
                    warmed += 1
                    del color_img

                    del depth_img
                else:
                    time.sleep(0.1) # 休眠时间和帧率有关系，计算方法：1/帧率，表示一帧需要的时间
            except Exception as e:
                logger.warning(f"Camera warm-up frame error: {e}")
                time.sleep(0.1)
            attempt += 1
        logger.info(f"Camera warm-up complete (got {warmed} valid frames in {attempt} attempts).")

    def reboot_camera(self):
        """
        供上位机主动调用的相机热重启接口 (不影响后台线程)
        常用于一个加工周期结束后，主动刷新 USB 链路，防止待机假死。
        """
        logger.info(">>> 主动触发相机硬件热重启 (Proactive Reboot) <<<")
        try:
            # 1. 安全断开底层句柄
            self.device.disconnect()

            # 2. 给操作系统 USB 驱动释放资源的喘息时间
            time.sleep(3.0)

            # 3. 重新连接
            success, msg = self.device.connect()
            if success:
                logger.info("相机热重启成功，执行预热...")
                self._warm_up()
                return True
            else:
                logger.error(f"相机热重启失败: {msg}。将在下次调用时重试。")
                return False

        except Exception as e:
            logger.error(f"相机热重启时发生异常: {e}")
            return False

    def _keep_alive_worker(self):
        while not self.stop_event.is_set():
            logger.info(f"Camera keep alive worker...")
            try:
                # 拔掉瞬间直接去取图会导致底层的 wait_for_frames 直接崩溃
                # if self.device.is_alive():
                #     ret, color_img, depth_img = self.device.get_frames(timeout_ms=500)
                #     if ret:
                #         logger.info(f"Camera keep alive worker, get img success")

                if not self.device.is_alive():
                    logger.info(f">>>>>> Connection lost in keep alive worker ")

                if not self.device.check_device_exist():
                    logger.warning("camera removed")

                if self.device.is_connected and not self.device.check_stream_responsive(timeout_ms=800):
                    self._keepalive_fail_count += 1
                    if self._keepalive_fail_count >= 3:
                        logger.warning("Keep alive: 3 次连续取帧超时，相机 RTP 流可能中断，深冲洗...")
                        self.device.flush_frames(num_frames=10)
                        self._keepalive_fail_count = 0
                else:
                    self._keepalive_fail_count = 0

            except Exception as e:
                logger.info(f"camera keep alive worker error: {e} \n {traceback.format_exc()}")
                pass

            time.sleep(30)


    def _save_worker(self):
        """后台存图线程函数"""
        logger.info("Image save worker started.")
        while not self.stop_event.is_set() or not self.save_queue.empty():
            try:
                # 设置超时以便能响应 stop_event
                item = self.save_queue.get(timeout=1.0)
                if item is None:  # 约定 None 为退出信号
                    break

                color_img, depth_img, timestamp = item

                # 确定保存路径
                date_str = time.strftime("%Y%m%d", time.localtime(timestamp / 1000))
                path = os.path.join(self.save_dir, date_str)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)

                if self.save_jpg:
                    # RGB -> BGR 并保存
                    bgr_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{path}/rgb_{timestamp}.jpg", bgr_img)
                    # 保存 16bit 深度图
                    cv2.imwrite(f"{path}/depth_{timestamp}.png", depth_img)

                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # print(f"Error in save worker: {e}")
                logger.error(f"Error in save worker: {e}")

    # def _save_to_local(self, color_arr, depth_arr):
    #     """保存图像到本地"""
    #     timestamp = int(time.time() * 1000)
    #     path = os.path.join(self.save_dir, time.strftime("%Y%m%d"))
    #     os.makedirs(path, exist_ok=True)
    #
    #     # RGB -> BGR for OpenCV
    #     bgr_img = cv2.cvtColor(color_arr, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f"{path}/rgb_{timestamp}.jpg", bgr_img)
    #     # 深度图保存为16位PNG
    #     cv2.imwrite(f"{path}/depth_{timestamp}.png", depth_arr)
    #     return f"{path}/rgb_{timestamp}.jpg"

    def execute_detection(self, ptype: int, detect: int=0, save_img: bool=True):
        """
        对外公开的同步业务接口
        :param ptype, 1-普通, 2-上料, 3-下料, 4-铝屑
        :param detect, 是否执行检测，0-只拍照，不检测，1-执行检测
        :return
            {
                "code": 0,    #  正常返回0，异常返回其他值
                "result": {
                    "ptype"：1， # 类型
                    "coords": [x,y,z,r], # 坐标参数
                    "ok": 1, # 检测结果，ok/1，ng/2
                    "exists": 1 # 根据ptype类型判断
                             1: exists == 1 表示有料，ok；
                             2: exists == 1 表示有料OK， 2表示空料ng；
                             3: exists == 2 表示空料；
                             4: exists == 1 有铁屑，表示ng；
                }
                "err_msg": ""   # 异常日志，0返回空
            }
        """
        last_err = ""
        consecutive_failures = 0
        recovery_failures = 0

        # 放大重试循环次数，给硬件重置留足空间
        MAX_RECOVERY_ATTEMPTS = 50

        # for attempt in range(self.max_retries):
        for attempt in range(MAX_RECOVERY_ATTEMPTS):
            # ========================================================
            # 阶段 1：物理连接与恢复
            # ========================================================

            if not self.device.is_alive():
                logger.info(f"Connection lost, retrying to connect (Attempt {attempt + 1})...")

                # 检查并尝试重连
                success, msg = self.device.connect()
                if not success:
                    last_err = msg
                    recovery_failures += 1
                    logger.warning(f"Connect failed. Recovery failure count: {recovery_failures}")

                    # 【核心修改】设备消失或连不上时，触发核弹重启
                    if recovery_failures >= 10:
                        logger.critical(">>> 连续 10 次无法连接设备，触发底层硬件重启 (Hardware Reboot) <<<")
                        if hasattr(self.device, 'hardware_reset'):
                            self.device.hardware_reset()
                        else:
                            self.device.disconnect()

                        logger.info("等待 6 秒让 USB 重新枚举...")
                        time.sleep(6.0)

                        # 重启后，重置一点故障数，给它机会走正常的 connect 流程
                        # 比如退回到 5，如果还是连不上，5次之后又会重启
                        recovery_failures = 5
                        continue

                    # 常规连接失败，睡 1 秒后重试
                    time.sleep(1.0)
                    continue
                else:
                    # 连接成功后，绝对不能立刻取图！必须给底层光学传感器 1.5 秒的预热时间，否则立刻取图会触发假掉帧
                    logger.info("Camera connected, warming up optical sensor...")
                    time.sleep(1.5)

                    # 预热后，把这 1.5 秒内积攒的废图抽掉，保证曝光正常
                    # if hasattr(self.device, 'flush_frames'):
                    self.device.flush_frames(num_frames=2)

            # ========================================================
            # 阶段 2：采集图像
            # ========================================================
            success, color_img, depth_img = self.device.get_frames(timeout_ms=2000)

            if not success:
                # consecutive_failures += 1

                recovery_failures += 1

                last_err = "Failed to capture frames"
                logger.warning(f"获取图像失败，当前连续失败次数: {recovery_failures}")
                # time.sleep(0.033)  # 等待约一帧的时间 (30fps的周期)
                time.sleep(0.1)  # 等待约一帧的时间 (10fps的周期)

                # get_frames失败，原因很多:
                #   - RTP UDP 超时/包乱序 (网口相机)
                #   - Align 失败/MJPG 解码失败/Depth Frame 为空
                #   - 只有连续大量失败时才真正去重启相机

                # 级别 1：轻度掉帧 (5次)，降速 + 深冲洗 RTP 缓存
                if recovery_failures == 5:
                    logger.warning(f"连续 {recovery_failures} 次掉帧，执行降速深冲洗...")
                    self.device.flush_frames(num_frames=30)
                    time.sleep(2.0)
                    continue

                # 级别 2：持续掉帧 (20次)，Pipeline 重建
                if recovery_failures == 20:
                    logger.warning(f">>> 连续 {recovery_failures} 次掉帧，执行 Pipeline 软重启 <<<")
                    self.device.disconnect()
                    time.sleep(2.0)
                    continue

                # 级别 3：极端掉帧 (40次)，硬件复位
                elif recovery_failures >= 40:
                    logger.critical(f"连续 {recovery_failures} 次掉帧，触发硬件重启机制！")
                    if hasattr(self.device, 'hardware_reset'):
                        self.device.hardware_reset()
                    else:
                        self.device.disconnect()

                    logger.info("等待相机主板重启并重新连接 (6秒)...")
                    time.sleep(6.0)

                    recovery_failures = 2
                    continue

                # 普通失败，进入下一轮重试
                continue

            # ========================================================
            # 阶段 3：成功拿到图，执行业务逻辑
            # ========================================================
            # 走到这里，说明物理层和流通道全通了，彻底清零故障累加器

            # consecutive_failures = 0
            recovery_failures = 0

            if not detect:
                logger.info(f"Detect denied ...")
                return {"code": 0, "result": {"ok": 1, "coords": [0, 0, 0, 0]}, "err_msg": ""}

            try:
                # 3. 将存图任务提交给后台队列 (非阻塞)
                timestamp = int(time.time() * 1000)
                try:
                    self.save_queue.put_nowait((color_img, depth_img, timestamp))
                except queue.Full:
                    logger.warning("Warning: Save queue full, dropping image.")
                except Exception as e:
                    logger.error(f"Error in image save queue: {e}")

                # 4. 调用算法检测
                result = self.detector.detect(ptype, color_img, depth_img)

                # 5. 绘图与展示
                if self.depth_show:
                    timestamp = int(time.time() * 1000)
                    date_str = time.strftime("%Y%m%d", time.localtime(timestamp / 1000))
                    path = os.path.join(self.save_dir, date_str)

                    depth_color = self.detector.depth_pseudo_color(depth_img)
                    result_img = self.detector.draw_result_with_rotated_box(depth_color, result)
                    # cv2.imshow("result-line", result_img)

                    # 缓存最新帧，供 HMI 界面读取
                    self._last_color_img = color_img.copy()
                    self._last_result_img = result_img.copy()

                    if self.save_jpg:
                    # if save_img:
                        cv2.imwrite(f"{path}/detect_result_horizontal_line_{timestamp}.jpg", result_img)

                    # cv2.waitKey(0)

                logger.info(f"detect result : {result}")
                detector_logger.info(f"detect result : {result}")

                return result

            except Exception as e:
                last_err = str(e)
                logger.error(f"Processing error: {e} \n traceback: {traceback.format_exc()}")
                # 发生未知错误，防呆断开相机
                self.device.disconnect()

        logger.info(f"Processing error: Max retries reached. {last_err}")
        return {"code": -1, "err_msg": f"Max retries reached. Last error: {last_err}"}

    def execute_detection_midian_depth(self, ptype:int, number:int=14, check_estop_func=None):
        """获取深度值的中位数返回值"""

        # 1. 执行一次带有重连保护的空调用，确保链路是通的，并顺便预热
        # 这一步保证了如果相机掉线，会在这里花 3 秒连上
        init_check = self.execute_detection(ptype, detect=0, save_img=False)
        if init_check.get("code") != 0:
            return {"code": -99, "err_msg": "Camera link broken before sequence"}

        # 在正式获取图像前，排空旧图
        # self.device.flush_frames(num_frames=5)

        # for idx in range(number):
        #     if idx < filter_number:  # 运动到点位之后立即拍照，图像深度不稳定，前7次只触发拍照，不处理
        #         self.execute_detection(ptype, detect=0)
        #         time.sleep(0.01)

        # 2. 预热排空 (非常重要，代替你之前的 if idx < 7: detect=0)
        # 直接让底层快速抽掉 7 帧，不经过任何上层封装，耗时仅需 0.2 秒！
        # filter_number = 9
        self.device.flush_frames(num_frames=const.disgard_frames_number)

        # required_count = number - filter_number
        consecutive_failures = 0

        results = []

        while len(results) < const.available_frames_number:
            if check_estop_func and check_estop_func():
                return {"code":-99, "err_msg":f"检测异常: 系统急停"}

            result = self.execute_detection(ptype, detect=1)  # 可以加上save_img=False节省时间
            # print(f"result is : {result}")

            if result["code"] == 0:
                consecutive_failures = 0  # 成功则重置失败计数

                # 过滤有效深度的值
                if ptype in (const.photo_type_loading, const.photo_type_find_head):
                    if result["result"]["coords"][2] <= const.depth_valid_filter:
                        results.append(result)
                else:
                    results.append(result)
            else:
                # 失败了，可能是偶发丢帧
                consecutive_failures += 1
                if consecutive_failures > 3:
                    logger.error("连拍过程中相机彻底掉线，提前终止中位数采集")
                    break

            # 给底层 USB 留出喘息时间，避免阻塞 (30fps = 33ms 一张)
            # time.sleep(0.01)

        # print("#########################################")

        if len(results) == 0:
            return {"code":-99, "err_msg":f"检测异常: 中位数处理返回空数组"}

        # 上下料，y, z取中位数，r取平均值
        if ptype in (const.photo_type_loading, const.photo_type_unloading, const.photo_type_find_head):
            n = len(results)
            mid = n // 2

            sorted_y_result = sorted(
                results,
                key=lambda x: x["result"]["coords"][1]  # x代表每个item，取z值（索引2）
            )
            y_result = [item['result']['coords'][1] for item in sorted_y_result]
            y_midian = y_result[mid] if n % 2 == 1 else (y_result[mid - 1] + y_result[mid]) / 2

            sorted_z_result = sorted(
                results,
                key=lambda x: x["result"]["coords"][2]  # x代表每个item，取z值（索引2）
            )
            z_result = [item['result']['coords'][2] for item in sorted_z_result]
            z_midian = z_result[mid] if n % 2 == 1 else (z_result[mid - 1] + z_result[mid]) / 2

            r_result = [item['result']['coords'][3] for item in sorted_z_result]
            r_average = sum(r_result) / len(r_result)
            r_midian = r_result[mid] if n % 2 == 1 else (r_result[mid - 1] + r_result[mid]) / 2

            midian_result = sorted_z_result[mid]
            midian_result["result"]["coords"][1] = y_midian
            midian_result["result"]["coords"][2] = z_midian
            midian_result["result"]["coords"][3] = r_midian

            with open(f"{get_logs_dir()}/detect_algo.log", "a+") as f:
                for item in sorted_z_result:
                    f.write(f"{item}\n")
                f.write(f"midian_result: {midian_result}\n")

            detector_logger.info(f"midian_result: {midian_result}")

            return midian_result

        return results[int(len(results)/2)]


    def shutdown(self):
        """释放资源"""
        logger.info("Shutting down service...")
        self.stop_event.set()
        self.save_queue.put(None)  # 发送退出信号
        self.save_thread.join(timeout=3.0)
        self.alive_thread.join(timeout=3.0)
        self.device.disconnect()

    def update_product(self, new_product_no):
        """切换产品型号，重新加载算法配置"""
        if self.product_no == new_product_no:
            return

        logger.info(f"Switching algorithm product to {new_product_no}...")
        self.product_no = new_product_no

        # 重新初始化 C++ 算法 (假设 C++ 有 reinit 接口，或者重新 new)
        # self.algo.initialize(self.product_no)

        self.detector.init(self.product_no)
        logger.info("Algorithm updated.")


def get_midian(results):
    n = len(results)
    mid = n // 2

    sorted_y_result = sorted(
        results,
        key=lambda x: x["result"]["coords"][1]  # x代表每个item，取z值（索引2）
    )
    print(f"sorted y result : {sorted_y_result}")
    y_result = [item['result']['coords'][1] for item in sorted_y_result]
    print(f"sorted y : {y_result}")
    y_midian = y_result[mid] if n % 2 == 1 else (y_result[mid - 1] + y_result[mid]) / 2
    print(f"y midian: {y_midian}")

    sorted_z_result = sorted(
        results,
        key=lambda x: x["result"]["coords"][2]  # x代表每个item，取z值（索引2）
    )
    z_result = [item['result']['coords'][2] for item in sorted_z_result]
    z_midian = z_result[mid] if n % 2 == 1 else (z_result[mid - 1] + z_result[mid]) / 2

    r_values = [item['result']['coords'][3] for item in sorted_z_result]
    r_average = sum(r_values) / len(r_values)

    midian_result = sorted_z_result[mid]
    midian_result["result"]["coords"][1] = y_midian
    midian_result["result"]["coords"][2] = z_midian
    midian_result["result"]["coords"][3] = r_average

    print(midian_result)

def main():
    # 初始化业务类
    service = DetectAlgoService(product_no="M001")
    service.depth_show = 1
    # time.sleep(3)

    try:
        # 上位机发起一次同步调用
        # ptype: 1 (物料识别)
        # for i in range(15):
        #     time.sleep(2)

        start_time = round(time.time() * 1000)
        logger.info(f"Start Time: {start_time}")
        # response = service.execute_detection(ptype=const.photo_type_loading, detect=1)
        # response = service.execute_detection_midian_depth(ptype=const.photo_type_loading)
        # response = service.execute_detection_midian_depth(ptype=const.photo_type_unloading)
        response = service.execute_detection_midian_depth(ptype=const.photo_type_find_head)
        # response = service.execute_detection_midian_depth(ptype=const.photo_type_normal)
        # response = service.execute_detection_midian_depth(ptype=const.photo_type_aluminum)
        # response = service.execute_detection_midian_depth(ptype=const.photo_type_cylinder)

        logger.info(f"response : {response}")
        print(response)

        # 处理结果
        if response["code"] == 0:
            res = response["result"]
            print(f"Detection OK: {response}")
        else:
            print(f"Detection Failed: {response['err_msg']}")
        end_time = round(time.time() * 1000)
        print(f"Detection Time: {end_time - start_time}")
        detector_logger.info(f"Detection Time: {end_time - start_time}")
    except Exception as e:
        logger.info(f"Detection Failed: {e}")

    finally:
        service.shutdown()


if __name__ == "__main__":
    main()