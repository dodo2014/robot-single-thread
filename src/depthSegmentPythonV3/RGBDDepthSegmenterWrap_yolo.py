import cv2
import numpy as np
import json
import os
import random
import traceback
from src.utils.path_helper import get_vision_detector_dir

# ===================== 常量定义 =====================
class SortRule:
    SORT_BY_Y_DESC = 0  # 按像素中心Y值 从大到小 ✅ 核心需求
    SORT_BY_Y_ASC = 1  # 按像素中心Y值 从小到大
    SORT_BY_X_DESC = 2  # 按像素中心X值 从大到小
    SORT_BY_X_ASC = 3  # 按像素中心X值 从小到大
    SORT_BY_AREA_DESC = 4  # 按区域面积 从大到小
    SORT_BY_DEPTH_ASC = 5  # 按平均深度 从小到大 (近→远)
    SORT_BY_ID_ASC = 6  # 按ID顺序


class PType:
    MATERIAL_RECOG = 1  # 普通拍照-物料识别
    FEED_EMPTY_CHECK = 2  # 上料-空料判断
    UNLOAD_FULL_CHECK = 3  # 下料-满料判断
    IRON_CHIP_CHECK = 4  # 铁屑识别


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
        # 深度区间过滤参数 ✅ 历史修改
        self.min_depth = 50
        self.max_depth = 2000
        # ROI感兴趣区域配置参数 ✅ 历史修改
        self.roi_x = 0
        self.roi_y = 0
        self.roi_w = 640
        self.roi_h = 480
        # ========== ✅ 修改+优化：YOLOv4铁屑检测 相关配置 ==========
        self.yolov4_cfg_path = "./yolov4/iron_chip.cfg"  # yolov4配置文件路径
        self.yolov4_weights_path = "./yolov4/iron_chip.weights"  # yolov4权重文件路径
        self.yolov4_names_path = "./yolov4/iron_chip.names"  # 类别名称文件路径
        self.yolov4_conf_threshold = 0.5  # 置信度阈值
        self.yolov4_nms_threshold = 0.45  # 非极大值抑制阈值
        self.yolov4_input_w = 608  # 输入宽度
        self.yolov4_input_h = 608  # 输入高度
        self.yolov4_net = None  # yolov4网络模型
        self.yolov4_classes = []  # 检测类别列表
        self.detect_iron_chips = []  # ✅ 修改：保存检测到的铁屑框坐标+置信度 [ (x1,y1,w,h,conf), ... ]

        # 末端工具相对相机中心的坐标
        self.tool_coord_x = 0
        self.tool_coord_y = 0
        self.tool_coord_z = 0
        self.tool_coord_r = 0

        # 8邻域偏移量
        self.neighbors_8 = [(-1, -1), (0, -1), (1, -1),
                            (-1, 0), (1, 0),
                            (-1, 1), (0, 1), (1, 1)]
        random.seed(10)
        # 保存分割后的区域结果，用于绘图接口调用
        self.detected_regions = []

    # ===================== 【私有函数】初始化YOLOv4模型加载 =====================
    def _init_yolov4_model(self):
        """加载YOLOv4模型和类别，只加载一次"""
        try:
            # 加载类别名称
            if os.path.exists(self.yolov4_names_path):
                with open(self.yolov4_names_path, 'r', encoding='utf-8') as f:
                    self.yolov4_classes = [line.strip() for line in f.readlines()]
            # 加载YOLOv4网络模型
            self.yolov4_net = cv2.dnn.readNetFromDarknet(self.yolov4_cfg_path, self.yolov4_weights_path)
            # 设置后端为CPU（如果有GPU可改为 cv2.dnn.DNN_BACKEND_CUDA + cv2.dnn.DNN_TARGET_CUDA）
            self.yolov4_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolov4_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("✅ YOLOv4铁屑检测模型加载成功！")
        except Exception as e:
            print(f"❌ YOLOv4模型加载失败: {str(e)} \n {traceback.format_exc()}")
            self.yolov4_net = None

    # ===================== 【私有函数】YOLOv4铁屑检测核心推理 ✅ 关键优化 =====================
    def _yolov4_detect_chip(self, rgb_img):
        """YOLOv4检测铁屑，返回是否检测到铁屑 + 铁屑框列表(带置信度)"""
        self.detect_iron_chips = []
        if self.yolov4_net is None or len(self.yolov4_classes) == 0:
            return False, []

        (H, W) = rgb_img.shape[:2]
        # 获取YOLOv4输出层
        ln = self.yolov4_net.getLayerNames()
        ln = [ln[i - 1] for i in self.yolov4_net.getUnconnectedOutLayers()]

        # 预处理图像：转为blob格式
        blob = cv2.dnn.blobFromImage(rgb_img, 1 / 255.0, (self.yolov4_input_w, self.yolov4_input_h), swapRB=True,
                                     crop=False)
        self.yolov4_net.setInput(blob)
        layer_outputs = self.yolov4_net.forward(ln)

        boxes = []
        confidences = []
        class_ids = []

        # 解析检测结果
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # 过滤置信度+只检测铁屑类别
                if confidence > self.yolov4_conf_threshold:  # and self.yolov4_classes[class_id] in ["aluminum_chip", "chip", "铁屑"]:
                    # 还原框坐标到原图尺寸
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 非极大值抑制去重
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.yolov4_conf_threshold, self.yolov4_nms_threshold)
        # ✅ 核心优化：保存 框坐标 + 置信度 双信息，用于绘图叠加显示
        if len(idxs) > 0:
            for i in idxs.flatten():
                x1, y1, w, h = boxes[i]
                conf = confidences[i]
                self.detect_iron_chips.append((x1, y1, w, h, round(conf, 2)))
        # 返回：是否检测到铁屑、铁屑框列表(带置信度)
        return len(self.detect_iron_chips) > 0, self.detect_iron_chips

    # ===================== 【对外接口1 - 初始化函数】=====================
    def init(self, product_no):
        try:
            self.product_no = product_no
            config_path = get_vision_detector_dir() / f"./config/{product_no}.json"
            if not os.path.exists(config_path):
                return {"code": -1, "err_msg": f"配置文件不存在: {config_path}"}

            # 读取json配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            # 加载所有原有配置参数
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
            # 加载深度区间过滤参数 ✅ 历史修改
            self.min_depth = cfg.get("min_depth", 50)
            self.max_depth = cfg.get("max_depth", 2000)
            # 加载ROI感兴趣区域参数 ✅ 历史修改
            self.roi_x = cfg.get("roi_x", 0)
            self.roi_y = cfg.get("roi_y", 0)
            self.roi_w = cfg.get("roi_w", 640)
            self.roi_h = cfg.get("roi_h", 480)
            # ========== ✅ 修改：加载YOLOv4相关配置参数 ==========
            self.yolov4_cfg_path = get_vision_detector_dir() / cfg.get("yolov4_cfg_path", "./yolov4/iron_chip.cfg")
            self.yolov4_weights_path = get_vision_detector_dir() / cfg.get("yolov4_weights_path", "./yolov4/iron_chip.weights")
            self.yolov4_names_path = get_vision_detector_dir() / cfg.get("yolov4_names_path", "./yolov4/iron_chip.names")
            self.yolov4_conf_threshold = cfg.get("yolov4_conf_threshold", 0.5)
            self.yolov4_nms_threshold = cfg.get("yolov4_nms_threshold", 0.45)
            self.yolov4_input_w = cfg.get("yolov4_input_w", 608)
            self.yolov4_input_h = cfg.get("yolov4_input_h", 608)

            # 末端工具相对相机中心的坐标
            self.tool_coord_x = cfg.get("tool_coord_x", 0)
            self.tool_coord_y = cfg.get("tool_coord_y", 0)
            self.tool_coord_z = cfg.get("tool_coord_z", 0)
            self.tool_coord_r = cfg.get("tool_coord_r", 0)

            self.config_loaded = True
            # 初始化YOLOv4模型
            self._init_yolov4_model()

            return {"code": 0}
        except Exception as e:
            return {"code": -2, "err_msg": f"初始化失败: {str(e)}"}

    # 像素坐标+深度 → XYZ真实三维坐标 (针孔相机模型)
    def _pixel2world(self, pixel, depth_mm):
        x = (pixel[0] - self.camera_cx) * depth_mm / self.camera_fx
        y = (pixel[1] - self.camera_cy) * depth_mm / self.camera_fy
        z = depth_mm
        return (round(x, 2), round(y, 2), round(z, 2))

    # ✅ 【核心新增函数】根据旋转矩形和排序规则，计算对应的边缘中心点
    def _get_edge_center_point(self, rotated_rect):
        """
        输入：旋转矩形 rotated_rect (cv2.minAreaRect返回的结果)
        返回：对应排序规则的边缘中心点 (x,y)
        SORT_BY_Y_DESC → 下边缘中心点 (Y最大的边的中点)
        SORT_BY_Y_ASC → 上边缘中心点 (Y最小的边的中点)
        其他规则 → 返回旋转矩形的中心点（兼容原逻辑）
        """
        # 获取旋转矩形的四个顶点
        box = cv2.boxPoints(rotated_rect)
        box = np.int32(np.round(box))
        # 提取四个顶点的y坐标，用于判断上下边缘
        ys = [p[1] for p in box]
        min_y = min(ys)
        max_y = max(ys)

        if self.sort_rule == SortRule.SORT_BY_Y_ASC:
            # 按Y升序 → 取上边缘(最小Y)的所有点，计算中点
            top_edge_points = [p for p in box if p[1] == min_y]
            cx = int((top_edge_points[0][0] + top_edge_points[1][0]) / 2)
            cy = int((top_edge_points[0][1] + top_edge_points[1][1]) / 2)
            return (cx, cy)
        elif self.sort_rule == SortRule.SORT_BY_Y_DESC:
            # 按Y降序 → 取下边缘(最大Y)的所有点，计算中点
            bottom_edge_points = [p for p in box if p[1] == max_y]
            cx = int((bottom_edge_points[0][0] + bottom_edge_points[1][0]) / 2)
            cy = int((bottom_edge_points[0][1] + bottom_edge_points[1][1]) / 2)
            return (cx, cy)
        else:
            # 其他排序规则 → 返回旋转矩形中心点，兼容原逻辑
            return (int(rotated_rect[0][0]), int(rotated_rect[0][1]))

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

    # 区域生长分割核心算法 - 全保留历史优化
    def _depth_segment(self, depth_img):
        regions = []
        h, w = depth_img.shape
        visited = np.zeros((h, w), dtype=np.uint8)
        depth_filtered = cv2.medianBlur(depth_img, self.median_blur_kernel)
        depth_filtered = cv2.GaussianBlur(depth_filtered, (3, 3), self.gaussian_sigma)
        region_count = 0

        # ROI区域边界值计算，防止越界
        roi_x_end = self.roi_x + self.roi_w
        roi_y_end = self.roi_y + self.roi_h
        roi_x_end = min(roi_x_end, w)
        roi_y_end = min(roi_y_end, h)
        self.roi_x = max(self.roi_x, 0)
        self.roi_y = max(self.roi_y, 0)

        # 遍历范围改为【仅ROI区域内】
        for y in range(self.roi_y, roi_y_end):
            for x in range(self.roi_x, roi_x_end):
                curr_depth_val = depth_filtered[y, x]
                # 五重过滤：无效值+已访问+最小深度+最大深度
                if curr_depth_val == self.depth_invalid or visited[y, x] == 1 or \
                        curr_depth_val < self.min_depth or curr_depth_val > self.max_depth:
                    continue

                seed_queue = [(x, y)]
                visited[y, x] = 1
                region_pixels = []
                # depth_sum 浮点型防溢出 ✅ 历史修复
                depth_sum = 0.0
                area = 0

                while seed_queue:
                    curr_x, curr_y = seed_queue.pop(0)
                    curr_depth = depth_filtered[curr_y, curr_x]
                    # 内层循环深度过滤
                    if curr_depth < self.min_depth or curr_depth > self.max_depth:
                        continue

                    region_pixels.append((curr_x, curr_y))
                    depth_sum += curr_depth
                    area += 1

                    for dx, dy in self.neighbors_8:
                        nx = curr_x + dx
                        ny = curr_y + dy
                        # ROI边界过滤
                        if nx < self.roi_x or nx >= roi_x_end or ny < self.roi_y or ny >= roi_y_end:
                            continue
                        if visited[ny, nx] == 1 or depth_filtered[ny, nx] == self.depth_invalid:
                            continue
                        if depth_filtered[ny, nx] < self.min_depth or depth_filtered[ny, nx] > self.max_depth:
                            continue

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
                # world_xyz = self._pixel2world(pixel_center, avg_depth)
                pts = np.array(region_pixels, dtype=np.int32).reshape((-1, 1, 2))
                rotated_rect = cv2.minAreaRect(pts)
                rotate_angle = round(rotated_rect[2], 2)

                # ==============================================
                # ✅ 核心修改点 START --- 世界坐标计算逻辑替换
                # ==============================================
                # 1. 调用新增函数，获取【边缘特征中心点】
                edge_center_point = self._get_edge_center_point(rotated_rect)
                # 2. 使用【边缘特征中心点】替代原像素中心点，计算真实世界XYZ坐标
                world_xyz = self._pixel2world(edge_center_point, avg_depth)
                # ==============================================
                # ✅ 核心修改点 END
                # ==============================================

                regions.append({
                    "region_id": region_count,
                    "pixel_center": pixel_center,  # 保留原中心点，用于排序
                    "edge_center_point": edge_center_point,  # 新增：边缘特征点，用于绘图可视化
                    "world_xyz": world_xyz,  # ✅ 新：边缘点计算的世界坐标
                    "rotate_angle": rotate_angle,
                    "rotated_rect": rotated_rect,
                    "area": area,
                    "avg_depth": avg_depth,
                    "color": (random.randint(40, 255), random.randint(40, 255), random.randint(40, 255))
                })
        self._sort_regions(regions)
        return regions

    # 检测逻辑判断
    def _judge_detect_result(self, regions, ptype, rgb_img):
        coords = [0.0, 0.0, 0.0, 0.0]
        ok_flag = DetectStatus.NG

        # ========== 核心逻辑 ✅ 铁屑检测 专用逻辑 ==========
        if ptype == PType.IRON_CHIP_CHECK:
            # 调用YOLOv4检测铁屑
            has_chip, chip_boxes = self._yolov4_detect_chip(rgb_img)
            if has_chip:
                ok_flag = DetectStatus.NG  # 检测到铁屑 → 结果NG
            else:
                ok_flag = DetectStatus.OK  # 未检测到铁屑 → 结果OK
            return ok_flag, coords

        # ========== 原有逻辑 完全不变 ==========
        if not regions:
            if ptype == PType.FEED_EMPTY_CHECK:
                ok_flag = DetectStatus.OK  # 空料判断-无区域=空料=OK
            return ok_flag, coords

        # 默认取排序后的第一个主区域作为目标
        main_region = regions[0]
        x, y, z = main_region["world_xyz"]
        r = main_region["rotate_angle"]
        x += self.tool_coord_x
        y += self.tool_coord_y
        z += self.tool_coord_z
        r += self.tool_coord_r
        coords = [x, y, z, r]
        area = main_region["area"]

        '''# 按检测类型做判断
        if ptype == PType.MATERIAL_RECOG:
            if self.detect_ok_area_min <= area <= self.detect_ok_area_max:
                ok_flag = DetectStatus.OK
        elif ptype == PType.FEED_EMPTY_CHECK:
            ok_flag = DetectStatus.NG # 有区域=非空=NG
        elif ptype == PType.UNLOAD_FULL_CHECK:
            if self.detect_ok_area_min <= area <= self.detect_ok_area_max:
                ok_flag = DetectStatus.OK # 有料且面积合规=满料=OK
        '''

        return ok_flag, coords

    # ===================== 【对外接口2 - 检测接口】核心 ✅ 传参修改 =====================
    def detect(self, ptype, rgb, depth):
        # 前置校验
        if not self.config_loaded:
            return {"code": -1, "result": {}, "err_msg": "请先调用初始化函数加载配置"}
        if depth is None or depth.dtype != np.uint16:
            return {"code": -2, "result": {}, "err_msg": "深度图格式错误，必须是np.uint16单通道"}
        if rgb is None:
            return {"code": -4, "result": {}, "err_msg": "RGB图不能为空，铁屑检测需要RGB图"}
        if ptype not in [1, 2, 3, 4]:
            return {"code": -3, "result": {}, "err_msg": "ptype类型错误，仅支持1/2/3/4"}

        try:
            regions = []
            # 只有非铁屑检测，才执行深度图分割
            if ptype != PType.IRON_CHIP_CHECK:
                regions = self._depth_segment(depth)
            self.detected_regions = regions  # 保存检测的区域，给绘图接口用

            # ========== 修改传参 ✅ 传入RGB图给判断逻辑，用于YOLOv4铁屑检测 ==========
            ok_flag, coords = self._judge_detect_result(regions, ptype, rgb)

            # 组装返回结果
            return {
                "code": 0,
                "result": {
                    "ptype": ptype,
                    "coords": coords,
                    "ok": ok_flag,
                    "exists": ok_flag
                },
                "err_msg": ""
            }
        except Exception as e:
            return {"code": -99, "result": {}, "err_msg": f"检测异常: {str(e)}"}

    # ===================== 【原有接口不变】叠加绘制函数 =====================
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

        # ✅ 修改：检测类型文本 铝屑识别 → 铁屑识别
        # type_dict = {1:"物料识别",2:"上料空料",3:"下料满料",4:"铁屑识别"}
        type_dict = {1: "Material Detect", 2: "Empty Feed", 3: "Full Unload", 4: "Iron Detect"}
        cv2.putText(draw_img, f"TYPE: {type_dict[ptype]}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 绘制坐标信息
        coord_text = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f} R:{r:.1f}"
        cv2.putText(draw_img, coord_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return draw_img

    # ===================== 【核心优化 ✅ 重点】叠加 YOLO铁屑框+置信度+旋转矩形框+中心点+ROI框+所有信息 =====================
    def draw_result_with_rotated_box(self, rgb, detect_res):
        draw_img = rgb.copy()
        # 检测异常的情况 加双重判断
        if detect_res["code"] != 0 or not detect_res["result"]:
            cv2.putText(draw_img, "DETECT ERR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if detect_res["err_msg"]:
                cv2.putText(draw_img, detect_res["err_msg"][:15], (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                            1)
            return draw_img

        res = detect_res["result"]
        ptype = res["ptype"]
        coords = res["coords"]
        ok_flag = res["ok"]
        x, y, z, r = coords

        # 1. 绘制原有所有信息：状态、类型、坐标
        status_text = "OK" if ok_flag == DetectStatus.OK else "NG"
        status_color = (0, 255, 0) if ok_flag == DetectStatus.OK else (0, 0, 255)
        cv2.putText(draw_img, f"STATUS: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        # ✅ 修改：检测类型文本 铝屑识别 → 铁屑识别
        # type_dict = {1:"物料识别",2:"上料空料",3:"下料满料",4:"铁屑识别"}
        type_dict = {1: "Material Detect", 2: "Empty Feed", 3: "Full Unload", 4: "Iron Detect"}
        cv2.putText(draw_img, f"TYPE: {type_dict[ptype]}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        coord_text = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f} R:{r:.1f}"
        cv2.putText(draw_img, coord_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 绘制ROI感兴趣区域框（浅蓝色 半透明）
        roi_x_end = self.roi_x + self.roi_w
        roi_y_end = self.roi_y + self.roi_h
        draw_img[self.roi_y:roi_y_end, self.roi_x:roi_x_end, 0] = np.clip(
            draw_img[self.roi_y:roi_y_end, self.roi_x:roi_x_end, 0] + 50, 0, 255)
        cv2.rectangle(draw_img, (self.roi_x, self.roi_y), (roi_x_end, roi_y_end), (255, 200, 0), 2)
        cv2.putText(draw_img, f"ROI", (self.roi_x + 5, self.roi_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0),
                    2)

        # ========== ✅ 核心新增+优化：铁屑检测专用绘制 - YOLO结果完整叠加 带置信度 醒目可视化 ==========
        if ptype == PType.IRON_CHIP_CHECK and len(self.detect_iron_chips) > 0:
            for (x1, y1, w, h, conf) in self.detect_iron_chips:
                # 绘制铁屑检测框 - 洋红色粗框 醒目
                cv2.rectangle(draw_img, (x1, y1), (x1 + w, y1 + h), (255, 0, 255), 2)
                # 绘制铁屑文本+置信度数值 带背景框防遮挡
                chip_text = f"IRON CHIP {conf}"
                cv2.rectangle(draw_img, (x1, y1 - 20), (x1 + len(chip_text) * 12, y1), (255, 0, 255), -1)
                cv2.putText(draw_img, chip_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 绘制目标旋转矩形框+中心点 (非铁屑检测时生效)
        if ptype != PType.IRON_CHIP_CHECK and len(self.detected_regions) > 0:
            main_region = self.detected_regions[0]
            pixel_center = main_region["pixel_center"]
            edge_center_point = main_region["edge_center_point"]  # 新增：边缘特征点
            rotated_rect = main_region["rotated_rect"]
            box_color = (0, 0, 255)
            # ✅ 历史修复：解决np.int0(box)报错
            box = cv2.boxPoints(rotated_rect)
            box = np.int32(np.round(box))
            cv2.drawContours(draw_img, [box], 0, box_color, 2)

            # ✅ 绘制 原中心点(黄色十字) + 边缘特征点(纯红色实心圆，醒目区分)
            cv2.drawMarker(draw_img, pixel_center, (0, 255, 255), cv2.MARKER_CROSS, markerSize=25, thickness=2)
            cv2.circle(draw_img, edge_center_point, 6, (0, 0, 255), -1)  # 实心红圈，代表世界坐标的计算点

            cv2.putText(draw_img, f"ID:{main_region['region_id']}", (pixel_center[0] + 10, pixel_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # ✅ 新增文本标注：边缘特征点的像素坐标
            cv2.putText(draw_img, f"Edge({edge_center_point[0]},{edge_center_point[1]})",
                        (edge_center_point[0] + 10, edge_center_point[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
    rgb_img = cv2.imread("./rgb_image.png")  # RGB图 - 铁屑检测必须
    depth_img = cv2.imread("./depth_image.png", cv2.IMREAD_UNCHANGED)  # 深度图

    if rgb_img is None or depth_img is None:
        print("读取图像失败，请检查路径！")
        exit(-1)

    # 兼容8位深度图转16位
    if depth_img.dtype == np.uint8:
        depth_img = depth_img.astype(np.uint16) * 20

    # 4. 调用检测接口 - 测试铁屑检测 ✅ 核心修改
    ptype = PType.IRON_CHIP_CHECK
    ptype = PType.MATERIAL_RECOG
    detect_res = detector.detect(ptype, rgb_img, depth_img)
    print("检测结果:\n", json.dumps(detect_res, ensure_ascii=False, indent=2))

    # 5. 绘制检测结果（带YOLO铁屑框+置信度+ROI+所有标注）
    result_img = detector.draw_result_with_rotated_box(rgb_img, detect_res)
    cv2.imshow("result", result_img)
    cv2.imwrite("./detect_result_iron_chip.jpg", result_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
