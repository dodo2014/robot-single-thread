import cv2
import numpy as np
import json
import os
import random

from src.utils.path_helper import get_vision_detector_dir

# ===================== 常量定义 =====================
# 排序规则枚举
class SortRule:
    SORT_BY_Y_DESC = 0  # 按像素中心Y值 从大到小 ✅ 核心需求
    SORT_BY_Y_ASC = 1  # 按像素中心Y值 从小到大
    SORT_BY_X_DESC = 2  # 按像素中心X值 从大到小
    SORT_BY_X_ASC = 3  # 按像素中心X值 从小到大
    SORT_BY_AREA_DESC = 4  # 按区域面积 从大到小
    SORT_BY_DEPTH_ASC = 5  # 按平均深度 从小到大 (近→远)
    SORT_BY_ID_ASC = 6  # 按ID顺序


# 检测类型枚举
class PType:
    MATERIAL_RECOG = 1  # 普通拍照-物料识别
    FEED_EMPTY_CHECK = 2  # 上料-空料判断
    UNLOAD_FULL_CHECK = 3  # 下料-满料判断
    ALUMINUM_CHIP_CHECK = 4  # 铝屑识别


# 结果状态枚举
class DetectStatus:
    OK = 1
    NG = 2


# ===================== 核心检测类 完整封装 =====================
class RGBDDetector:
    def __init__(self):
        # 初始化配置参数
        self.config_loaded = False
        self.product_no = ""
        self.camera_fx = 615.0
        self.camera_fy = 615.0
        self.camera_cx = 320.0
        self.camera_cy = 240.0
        self.depth_invalid = 0
        self.depth_diff_thresh = 35
        self.min_region_area = 150
        self.median_blur_kernel = 3
        self.gaussian_sigma = 1.2
        self.sort_rule = SortRule.SORT_BY_Y_DESC
        self.detect_ok_area_min = 500
        self.detect_ok_area_max = 10000

        # 8邻域偏移量
        self.neighbors_8 = [(-1, -1), (0, -1), (1, -1),
                            (-1, 0), (1, 0),
                            (-1, 1), (0, 1), (1, 1)]
        random.seed(10)

    # ===================== 【对外接口1 - 初始化函数】=====================
    # 说明: 初始化算法配置，根据物料编号读取配置文件
    # 参数: product_no - 物料编号
    # 返回: {"code":0} 0=成功
    def init(self, product_no):
        try:
            self.product_no = product_no
            # config_path = f"./config/{product_no}.json"
            config_path = get_vision_detector_dir() / f"./config/{product_no}.json"
            if not os.path.exists(config_path):
                return {"code": -1, "err_msg": f"配置文件不存在: {config_path}"}

            # 读取json配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            # 加载所有配置参数
            self.camera_fx = cfg.get("camera_fx", 615.0)
            self.camera_fy = cfg.get("camera_fy", 615.0)
            self.camera_cx = cfg.get("camera_cx", 320.0)
            self.camera_cy = cfg.get("camera_cy", 240.0)
            self.depth_invalid = cfg.get("depth_invalid", 0)
            self.depth_diff_thresh = cfg.get("depth_diff_thresh", 35)
            self.min_region_area = cfg.get("min_region_area", 150)
            self.median_blur_kernel = cfg.get("median_blur_kernel", 3)
            self.gaussian_sigma = cfg.get("gaussian_sigma", 1.2)
            self.sort_rule = cfg.get("sort_rule", 0)
            self.detect_ok_area_min = cfg.get("detect_ok_area_min", 500)
            self.detect_ok_area_max = cfg.get("detect_ok_area_max", 10000)

            self.config_loaded = True
            return {"code": 0}
        except Exception as e:
            return {"code": -2, "err_msg": f"初始化失败: {str(e)}"}

    # 像素坐标+深度 → XYZ真实三维坐标 (针孔相机模型)
    def _pixel2world(self, pixel, depth_mm):
        x = (pixel[0] - self.camera_cx) * depth_mm / self.camera_fx
        y = (pixel[1] - self.camera_cy) * depth_mm / self.camera_fy
        z = depth_mm
        return (round(x, 2), round(y, 2), round(z, 2))

    # 区域排序核心函数
    def _sort_regions(self, regions):
        if not regions:
            return
        if self.sort_rule == SortRule.SORT_BY_Y_DESC:
            regions.sort(key=lambda x: x["pixel_center"][1], reverse=True)
        elif self.sort_rule == SortRule.SORT_BY_Y_ASC:
            regions.sort(key=lambda x: x["pixel_center"][1])
        elif self.sort_rule == SortRule.SORT_BY_X_DESC:
            regions.sort(key=lambda x: x["pixel_center"][0], reverse=True)
        elif self.sort_rule == SortRule.SORT_BY_X_ASC:
            regions.sort(key=lambda x: x["pixel_center"][0])
        elif self.sort_rule == SortRule.SORT_BY_AREA_DESC:
            regions.sort(key=lambda x: x["area"], reverse=True)
        elif self.sort_rule == SortRule.SORT_BY_DEPTH_ASC:
            regions.sort(key=lambda x: x["avg_depth"])
        # 更新ID
        for i in range(len(regions)):
            regions[i]["region_id"] = i + 1

    # 区域生长分割核心算法
    def _depth_segment(self, depth_img):
        regions = []
        h, w = depth_img.shape
        visited = np.zeros((h, w), dtype=np.uint8)
        depth_filtered = cv2.medianBlur(depth_img, self.median_blur_kernel)
        depth_filtered = cv2.GaussianBlur(depth_filtered, (3, 3), self.gaussian_sigma)
        region_count = 0

        for y in range(h):
            for x in range(w):
                if depth_filtered[y, x] == self.depth_invalid or visited[y, x] == 1:
                    continue

                seed_queue = [(x, y)]
                visited[y, x] = 1
                region_pixels = []
                depth_sum = 0
                area = 0

                while seed_queue:
                    curr_x, curr_y = seed_queue.pop(0)
                    curr_depth = depth_filtered[curr_y, curr_x]
                    region_pixels.append((curr_x, curr_y))
                    depth_sum += curr_depth
                    area += 1

                    for dx, dy in self.neighbors_8:
                        nx = curr_x + dx
                        ny = curr_y + dy
                        if nx < 0 or nx >= w or ny < 0 or ny >= h: continue
                        if visited[ny, nx] == 1 or depth_filtered[ny, nx] == self.depth_invalid: continue
                        depth_diff = abs(int(depth_filtered[ny, nx]) - int(curr_depth))
                        if depth_diff <= self.depth_diff_thresh:
                            visited[ny, nx] = 1
                            seed_queue.append((nx, ny))

                if area < self.min_region_area:
                    continue

                region_count += 1
                avg_depth = int(depth_sum / area)
                cx = int(np.mean([p[0] for p in region_pixels]))
                cy = int(np.mean([p[1] for p in region_pixels]))
                pixel_center = (cx, cy)
                world_xyz = self._pixel2world(pixel_center, avg_depth)
                pts = np.array(region_pixels, dtype=np.int32).reshape((-1, 1, 2))
                rotated_rect = cv2.minAreaRect(pts)
                rotate_angle = round(rotated_rect[2], 2)

                regions.append({
                    "region_id": region_count,
                    "pixel_center": pixel_center,
                    "world_xyz": world_xyz,
                    "rotate_angle": rotate_angle,
                    "rotated_rect": rotated_rect,
                    "area": area,
                    "avg_depth": avg_depth,
                    "color": (random.randint(40, 255), random.randint(40, 255), random.randint(40, 255))
                })
        self._sort_regions(regions)
        return regions

    # 检测逻辑判断
    def _judge_detect_result(self, regions, ptype):
        coords = [0.0, 0.0, 0.0, 0.0]
        ok_flag = DetectStatus.NG

        if not regions:
            if ptype == PType.FEED_EMPTY_CHECK:
                ok_flag = DetectStatus.OK  # 空料判断-无区域=空料=OK
            return ok_flag, coords

        # 默认取排序后的第一个主区域作为目标
        main_region = regions[0]
        x, y, z = main_region["world_xyz"]
        r = main_region["rotate_angle"]
        coords = [x, y, z, r]
        area = main_region["area"]

        # 按检测类型做判断
        if ptype == PType.MATERIAL_RECOG:
            if self.detect_ok_area_min <= area <= self.detect_ok_area_max:
                ok_flag = DetectStatus.OK
        elif ptype == PType.FEED_EMPTY_CHECK:
            ok_flag = DetectStatus.NG  # 有区域=非空=NG
        elif ptype == PType.UNLOAD_FULL_CHECK:
            if self.detect_ok_area_min <= area <= self.detect_ok_area_max:
                ok_flag = DetectStatus.OK  # 有料且面积合规=满料=OK
        elif ptype == PType.ALUMINUM_CHIP_CHECK:
            if area > self.min_region_area:
                ok_flag = DetectStatus.NG  # 有铝屑=NG

        return ok_flag, coords

    # ===================== 【对外接口2 - 检测接口】核心 ✅ =====================
    # 说明: 物料检测，包括检测和定位坐标输出，纯算法无绘图
    # 参数: ptype - 检测类型 1/2/3/4; rgb - rgb图像(暂用，预留); depth - 深度图(np.uint16)
    # 返回: 严格指定格式的字典结果
    def detect(self, ptype, rgb, depth):
        # 前置校验
        if not self.config_loaded:
            return {"code": -1, "result": {}, "err_msg": "请先调用初始化函数加载配置"}
        if depth is None or depth.dtype != np.uint16:
            return {"code": -2, "result": {}, "err_msg": "深度图格式错误，必须是np.uint16单通道"}
        if ptype not in [1, 2, 3, 4]:
            return {"code": -3, "result": {}, "err_msg": "ptype类型错误，仅支持1/2/3/4"}

        try:
            # 执行分割+检测判断
            regions = self._depth_segment(depth)
            ok_flag, coords = self._judge_detect_result(regions, ptype)

            # 组装返回结果
            return {
                "code": 0,
                "result": {
                    "ptype": ptype,
                    "coords": coords,
                    "ok": ok_flag
                },
                "err_msg": ""
            }
        except Exception as e:
            return {"code": -99, "result": {}, "err_msg": f"检测异常: {str(e)}"}

    # ===================== 【对外接口3 - 叠加函数】=====================
    # 说明: 单独的结果叠加绘制函数，将检测结果绘制到图像上返回
    # 参数: rgb - 原始rgb图像; detect_res - detect接口返回的结果字典
    # 返回: 叠加了检测结果的彩色图像
    def draw_result(self, rgb, detect_res):
        draw_img = rgb.copy()
        if detect_res["code"] != 0 or not detect_res["result"]:
            cv2.putText(draw_img, "DETECT ERR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return draw_img

        res = detect_res["result"]
        ptype = res["ptype"]
        coords = res["coords"]
        ok_flag = res["ok"]
        x, y, z, r = coords

        # 绘制状态文字
        status_text = "OK" if ok_flag == DetectStatus.OK else "NG"
        status_color = (0, 255, 0) if ok_flag == DetectStatus.OK else (0, 0, 255)
        cv2.putText(draw_img, f"STATUS: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # 绘制检测类型
        # type_dict = {1:"物料识别",2:"上料空料",3:"下料满料",4:"铝屑识别"}
        type_dict = {1: "Material Detect", 2: "Empty Feed", 3: "Full Unload", 4: "Aluminum Chip Detect"}
        cv2.putText(draw_img, f"TYPE: {type_dict[ptype]}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 绘制坐标信息
        coord_text = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f} R:{r:.1f}"  # °
        cv2.putText(draw_img, coord_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return draw_img


# ===================== 测试主函数 调用示例 =====================
if __name__ == "__main__":
    # 1. 创建检测器实例
    detector = RGBDDetector()

    # 2. 【调用初始化接口】传入物料编号
    product_no = "M001"
    init_res = detector.init(product_no)
    if init_res["code"] != 0:
        print(f"初始化失败: {init_res['err_msg']}")
        exit(-1)
    print("初始化成功！")

    # 3. 读取图像
    rgb_img = cv2.imread("./rgb_image.png")  # RGB图
    depth_img = cv2.imread("./depth_image.png", cv2.IMREAD_UNCHANGED)  # 深度图必须IMREAD_UNCHANGED

    # 8位转16
    if depth_img.dtype == np.uint8:
        scale = 5120 / 256
        depth_img = np.uint16(depth_img * scale)

    # 4. 【调用检测接口】指定检测类型 1=物料识别
    ptype = 1
    detect_res = detector.detect(ptype, rgb_img, depth_img)
    print("检测结果:\n", json.dumps(detect_res, ensure_ascii=False, indent=2))

    # 5. 【调用叠加函数】绘制结果到RGB图
    if detect_res["code"] == 0:
        result_img = detector.draw_result(rgb_img, detect_res)
        cv2.imshow("检测结果叠加图", result_img)
        cv2.imwrite("./detect_result.png", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
