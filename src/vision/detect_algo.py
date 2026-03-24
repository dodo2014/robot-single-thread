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
        self.save_queue = queue.Queue(maxsize=100)  # 限制队列长度防止内存溢出
        self.stop_event = threading.Event()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

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
        """抽取出预热逻辑"""
        # 抛弃前 10-20 帧，让自动曝光稳定，并强制触发 SW_MODE 的查找表计算
        logger.info("Warming up camera...")
        try:
            for _ in range(20):
                self.device.get_frames(timeout_ms=100)
            logger.info("Camera is ready.")
        except Exception as e:
            logger.warning(f"Camera warm-up interrupted: {e}")

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

                # RGB -> BGR 并保存
                bgr_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{path}/rgb_{timestamp}.jpg", bgr_img)
                # 保存 16bit 深度图
                cv2.imwrite(f"{path}/depth_{timestamp}.png", depth_img)

                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in save worker: {e}")

    def _save_to_local(self, color_arr, depth_arr):
        """保存图像到本地"""
        timestamp = int(time.time() * 1000)
        path = os.path.join(self.save_dir, time.strftime("%Y%m%d"))
        os.makedirs(path, exist_ok=True)

        # RGB -> BGR for OpenCV
        bgr_img = cv2.cvtColor(color_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{path}/rgb_{timestamp}.jpg", bgr_img)
        # 深度图保存为16位PNG
        cv2.imwrite(f"{path}/depth_{timestamp}.png", depth_arr)
        return f"{path}/rgb_{timestamp}.jpg"

    def execute_detection(self, ptype: int, detect: int=0):
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

        for attempt in range(self.max_retries):
            # 1. 检查并尝试重连
            if not self.device.is_alive():
                logger.info(f"Connection lost, retrying to connect (Attempt {attempt + 1})...")
                success, msg = self.device.connect()
                if not success:
                    last_err = msg
                    time.sleep(1)
                    continue

            # 2. 采集图像
            success, color_frame, depth_frame = self.device.get_frames()
            if not success:
                last_err = "Failed to capture frames"
                time.sleep(0.033)  # 等待约一帧的时间 (30fps的周期)
                # 只有当连续多次（例如超过3次）都拿不到图时，才真正去重启相机
                if attempt >= 3:
                    logger.warning("Multiple consecutive frame drops, re-initializing pipeline...")
                    self.device.connect()
                # # 采集失败通常意味着链路抖动，尝试重新初始化 pipeline
                # self.device.connect()
                continue

            if not detect:
                return {"code": 0, "result": {"ok": 1, "coords": [0, 0, 0, 0]}, "err_msg": ""}

            try:
                # 彩色图转换: RGB888 每个像素 3 字节 (uint8)
                # color_frame.get_data() 是原始 buffer
                # color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                # color_img = color_data.reshape((720, 1280, 3))
                # color_img = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(
                #     (self.height, self.width, self.channel)).copy()

                f_width = color_frame.get_width()  # 动态获取当前帧的宽度
                f_height = color_frame.get_height()  # 动态获取当前帧的高度

                color_format = color_frame.get_format()
                raw_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

                if color_format == OBFormat.MJPG:
                    # 如果是 MJPG，使用 OpenCV 解码成 BGR
                    bgr_img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
                    if bgr_img is None:
                        raise ValueError("MJPG decode failed")
                    # 转换为 RGB (因为你之前的逻辑是存图前转 BGR，或者算法需要 RGB)
                    color_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                elif color_format == OBFormat.RGB:
                    # 只有格式确实是 RGB 时才能直接 reshape
                    color_img = raw_data.reshape((f_height, f_width, 3)).copy()
                else:
                    # 处理其他可能的格式（如 YUYV）
                    # 这里建议打印一下当前的格式，方便调试
                    print(f"Unsupported format for direct reshape: {color_format}")
                    logger.error(f"Unsupported format: {color_format}")
                    # 这种情况下通常需要专门的转换函数
                    return {"code": -1, "err_msg": f"Unsupported format {color_format}"}

                # 深度图转换: Y16 每个像素 2 字节 (uint16)
                # 使用 np.frombuffer 并指定 dtype=np.uint16
                # depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                # depth_img = depth_data.reshape((f_height, f_width)).copy()

                # depth_img = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                #     (f_height, f_width)).copy()

                d_height = depth_frame.get_height()  # 获取深度帧实际高度
                d_width = depth_frame.get_width()  # 获取深度帧实际宽度

                depth_img = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (d_height, d_width)).copy()

                # 本地持久化
                # self._save_to_local(color_img, depth_img)

                # 2. 将存图任务提交给后台队列 (非阻塞)
                timestamp = int(time.time() * 1000)
                try:
                    self.save_queue.put_nowait((color_img, depth_img, timestamp))
                except queue.Full:
                    logger.warning("Warning: Save queue full, dropping image.")
                except Exception as e:
                    logger.error(f"Error in image save queue: {e}")

                # 5. 二进制处理 (传递给 C++ 算法)
                # 直接获取原始内存 Buffer 的 bytes 形式
                # rgb_binary = color_frame.get_data().tobytes()
                # depth_binary = depth_frame.get_data().tobytes()

                # 6. 调用 C++ 算法
                # result = self.algo.detect(ptype, rgb_binary, depth_binary)
                # return result

                result = self.detector.detect(ptype, color_img, depth_img)

                # logger.info(f"detector result: {result}")

                if self.depth_show:
                    timestamp = int(time.time() * 1000)
                    date_str = time.strftime("%Y%m%d", time.localtime(timestamp / 1000))
                    path = os.path.join(self.save_dir, date_str)

                    depth_color = self.detector.depth_pseudo_color(depth_img)
                    result_img = self.detector.draw_result_with_rotated_box(depth_color, result)
                    # cv2.imshow("result-line", result_img)

                    cv2.imwrite(f"{path}/detect_result_horizontal_line_{timestamp}.jpg", result_img)

                    cv2.waitKey(0)

                logger.info(f"detect result : {result}")

                # 显式释放底层 C++ 帧缓冲！！非常重要！！
                # 让 pyorbbecsdk 立即将 Buffer 归还给 SDK，防止缓存池枯竭
                del color_frame
                del depth_frame

                return result

                # return {"code": 0, "result": {"ok": 1, "coords": [0, 0, 0, 0]}, "err_msg": ""}

            except Exception as e:
                last_err = str(e)
                print(f"Error in image save queue: {traceback.format_exc()}")
                logger.error(f"Processing error: {e} \n traceback: {traceback.format_exc()}")
                # 如果发生异常，确保释放帧，防止内存泄漏
                if 'color_frame' in locals(): del color_frame
                if 'depth_frame' in locals(): del depth_frame

        logger.info(f"Processing error: Max retries reached. {last_err}")
        return {"code": -1, "err_msg": f"Max retries reached. Last error: {last_err}"}

    def execute_detection_midian_depth(self, ptype:int, number:int=21):
        """获取深度值的中位数返回值"""
        results = []
        for idx in range(number):
            if idx < 7:  # 运动到点位之后立即拍照，图像深度不稳定，前7次只触发拍照，不处理
                self.execute_detection(ptype, idx)
            else:
                result = self.execute_detection(ptype, detect=1)
                print(f"result is : {result}")


                if result["code"] == 0:
                    results.append(result)

            # 给底层 USB 留出喘息时间，避免阻塞 (30fps = 33ms 一张)
            time.sleep(0.01)

        print("#########################################")

        if len(results) == 0:
            return {"code":-99, "err_msg":f"检测异常: 中位数处理返回空数组"}

        if ptype in (const.photo_type_loading, const.photo_type_unloading):
            sorted_result = sorted(
                results,
                key=lambda x: x["result"]["coords"][2]  # x代表每个item，取z值（索引2）
            )
            print(f"sorted result len :{len(sorted_result)}, result: {sorted_result}")

            z_values = [item['result']['coords'][2] for item in sorted_result]
            z_average = sum(z_values) / len(z_values)

            r_values = [item['result']['coords'][3] for item in sorted_result]
            r_average = sum(r_values) / len(r_values)

            midian_result = sorted_result[int(len(sorted_result)/2)]
            midian_result["result"]["coords"][3] = r_average
            # midian_result["result"]["coords"][2] = z_average

            print(int(len(sorted_result)/2))
            print(get_logs_dir())
            with open(f"{get_logs_dir()}/detect_algo.log", "a+") as f:
                for item in sorted_result:
                    f.write(f"{item}\n")
                f.write(f"midian_result: {midian_result}\n")


            return midian_result

        return results[int(len(results)/2)]


    def shutdown(self):
        """释放资源"""
        logger.info("Shutting down service...")
        self.stop_event.set()
        self.save_queue.put(None)  # 发送退出信号
        self.save_thread.join(timeout=3.0)
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
        # response = service.execute_detection(ptype=2)
        response = service.execute_detection_midian_depth(ptype=2)
        print(response)

        # 处理结果
        if response["code"] == 0:
            res = response["result"]
            print(f"Detection OK: {response}")
        else:
            print(f"Detection Failed: {response['err_msg']}")
        end_time = round(time.time() * 1000)
        print(f"Detection Time: {end_time - start_time}")
        with open(f"{get_logs_dir()}/detect_algo.log", "a+") as f:
            f.write(f"Detection Time: {end_time - start_time}\n")

    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
    # dev = OrbbecCameraDevice()
    # success, msg = dev.connect()
    # if success:
    #     print("相机连接成功！")
    #     ret, color, depth = dev.get_frames()
    #     if ret:
    #         print("获取图像帧成功！")
    #     dev.disconnect()
    # else:
    #     print(f"连接失败: {msg}")
