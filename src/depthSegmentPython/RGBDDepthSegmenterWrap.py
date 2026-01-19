import cv2
import numpy as np
import json
import os
import random
import re
from src.utils.path_helper import get_vision_detector_dir

# ===================== 常量定义 =====================
class SortRule:
    SORT_BY_Y_DESC = 0  # 按Y降序 → 从ROI底部向上扫描，找第一个水平直线 ✅核心联动
    SORT_BY_Y_ASC = 1  # 按Y升序 → 从ROI顶部向下扫描，找第一个水平直线 ✅核心联动
    SORT_BY_X_DESC = 2
    SORT_BY_X_ASC = 3
    SORT_BY_AREA_DESC = 4
    SORT_BY_DEPTH_ASC = 5
    SORT_BY_ID_ASC = 6


class PType:
    MATERIAL_CHECK = 1
    FEED_CHECK = 2
    UNLOAD_CHECK = 3
    IRON_CHIP_CHECK = 4


class DetectStatus:
    UNKNOWN = 0
    OK = 1
    NG = 2
    EXIST = 1
    NOTHING = 2


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
        self.median_blur_kernel = 3
        self.gaussian_sigma = 1.2
        self.sort_rule = SortRule.SORT_BY_Y_DESC
        # 深度区间过滤参数
        self.feed_depth_min = 50
        self.feed_depth_max = 2000

        # ===================== 多ROI配置拆分 =====================
        # 1. 上料ROI（原ROI）
        self.feed_roi_x = 0
        self.feed_roi_y = 0
        self.feed_roi_w = 640
        self.feed_roi_h = 480
        # 2. 物料缓存台ROI + 深度范围
        self.material_roi_x = 0
        self.material_roi_y = 0
        self.material_roi_w = 640
        self.material_roi_h = 480
        self.material_depth_min = 100  # 物料存在的最小深度
        self.material_depth_max = 500  # 物料存在的最大深度
        # 3. YOLO铁屑检测ROI
        self.yolo_roi_x = 0
        self.yolo_roi_y = 0
        self.yolo_roi_w = 640
        self.yolo_roi_h = 480

        # YOLO铁屑检测 相关配置
        self.yolov4_cfg_path = "./yolov4/iron_chip.cfg"  # yolov4配置文件路径
        self.yolov4_weights_path = "./yolov4/iron_chip.weights"  # yolov4权重文件路径
        self.yolov4_names_path = "./yolov4/iron_chip.names"  # 类别名称文件路径
        self.yolov4_conf_threshold = 0.5
        self.yolov4_nms_threshold = 0.45
        self.yolov4_input_w = 608  # 输入宽度
        self.yolov4_input_h = 608  # 输入高度
        self.yolov4_net = None
        self.yolov4_classes = []
        self.detect_iron_chips = []
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
        # ===================== 新增：水平直线判定阈值（可按需微调） =====================
        self.horizontal_y_var_thresh = 3  # Y坐标方差阈值 <3 → 接近水平，越小越严格
        self.horizontal_x_span_min = 30  # X坐标最小跨度 >30 → 有效直线，过滤孤立点
        self.horizontal_pixel_min = 20  # 最小像素数 >20 → 过滤小毛刺

        # ===================== 下料算法配置参数 =====================
        self.unload_roi_x = 0  # 下料区域ROI X
        self.unload_roi_y = 0  # 下料区域ROI Y
        self.unload_roi_w = 640  # 下料区域ROI 宽度
        self.unload_roi_h = 480  # 下料区域ROI 高度
        self.unload_layer_count = 3  # 下料层数
        self.unload_layer_height = 50  # 层高（mm）
        self.unload_item_count_per_layer = 5  # 每层放置产品数量
        self.unload_item_interval = 80  # 产品间隔（像素/物理尺寸，根据标定转换）
        self.unload_item_width = 60  # 产品宽度（像素）
        self.unload_depth_threshold = 20  # 深度差阈值：判断该位置是否有物料

    # ===================== 【私有函数】初始化YOLOv4模型加载 =====================
    def _init_yolov4_model(self):
        try:
            if os.path.exists(self.yolov4_names_path):
                with open(self.yolov4_names_path, 'r', encoding='utf-8') as f:
                    self.yolov4_classes = [line.strip() for line in f.readlines()]
            self.yolov4_net = cv2.dnn.readNetFromDarknet(self.yolov4_cfg_path, self.yolov4_weights_path)
            self.yolov4_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolov4_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("✅ YOLOv4铁屑检测模型加载成功！")
        except Exception as e:
            print(f"❌ YOLOv4模型加载失败: {str(e)}")
            self.yolov4_net = None

    # ===================== 【私有函数】YOLOv4铁屑检测核心推理 ✅ 新增ROI过滤 =====================
    def _yolov4_detect_chip(self, rgb_img):
        """YOLOv4检测铁屑，返回是否检测到铁屑 + 铁屑框列表(带置信度)，新增ROI过滤"""
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
                if confidence > self.yolov4_conf_threshold:
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

        # ✅ 新增：YOLO检测结果ROI过滤（质心不在ROI内则删除）
        yolo_roi_x_end = self.yolo_roi_x + self.yolo_roi_w
        yolo_roi_y_end = self.yolo_roi_y + self.yolo_roi_h
        # 限制ROI范围在图像内
        yolo_roi_x = max(self.yolo_roi_x, 0)
        yolo_roi_y = max(self.yolo_roi_y, 0)
        yolo_roi_x_end = min(yolo_roi_x_end, W)
        yolo_roi_y_end = min(yolo_roi_y_end, H)

        # ✅ 核心优化：保存 框坐标 + 置信度 双信息，同时过滤ROI外的结果
        if len(idxs) > 0:
            for i in idxs.flatten():
                x1, y1, w, h = boxes[i]
                conf = confidences[i]
                # 计算检测框质心
                centroid_x = x1 + w // 2
                centroid_y = y1 + h // 2
                # 检查质心是否在YOLO ROI内
                if (yolo_roi_x <= centroid_x < yolo_roi_x_end and
                        yolo_roi_y <= centroid_y < yolo_roi_y_end):
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

            with open(config_path, 'r', encoding='utf-8') as f:
                # cfg = json.load(f)
                json_str = f.read()
                # 正则1：删除 /* 任意多行注释 */
                json_str = re.sub(r"/\*[\s\S]*?\*/", "", json_str)
                # 正则2：删除 // 单行注释 (包括行尾注释)
                json_str = re.sub(r"//.*", "", json_str)
                # 正则3：删除多余空行（可选，优化格式）
                json_str = re.sub(r"\n+", "\n", json_str).strip()
                # 解析清理后的纯JSON内容
                cfg = json.loads(json_str)

            # 基础相机参数
            self.camera_fx = cfg.get("camera_fx", 615.0)
            self.camera_fy = cfg.get("camera_fy", 615.0)
            self.camera_cx = cfg.get("camera_cx", 320.0)
            self.camera_cy = cfg.get("camera_cy", 240.0)
            self.depth_invalid = cfg.get("depth_invalid", 0)
            self.median_blur_kernel = cfg.get("median_blur_kernel", 3)
            self.gaussian_sigma = cfg.get("gaussian_sigma", 1.2)
            self.sort_rule = cfg.get("sort_rule", 0)
            self.feed_depth_min = cfg.get("feed_depth_min", 50)
            self.feed_depth_max = cfg.get("feed_depth_max", 2000)

            # ===================== 多ROI配置加载 =====================
            # 1. 上料ROI
            self.feed_roi_x = cfg.get("feed_roi_x", 0)
            self.feed_roi_y = cfg.get("feed_roi_y", 0)
            self.feed_roi_w = cfg.get("feed_roi_w", 640)
            self.feed_roi_h = cfg.get("feed_roi_h", 480)
            # 2. 物料缓存台ROI + 深度范围
            self.material_roi_x = cfg.get("material_roi_x", 0)
            self.material_roi_y = cfg.get("material_roi_y", 0)
            self.material_roi_w = cfg.get("material_roi_w", 640)
            self.material_roi_h = cfg.get("material_roi_h", 480)
            self.material_depth_min = cfg.get("material_depth_min", 100)
            self.material_depth_max = cfg.get("material_depth_max", 500)
            # 3. YOLO铁屑检测ROI
            self.yolo_roi_x = cfg.get("yolo_roi_x", 0)
            self.yolo_roi_y = cfg.get("yolo_roi_y", 0)
            self.yolo_roi_w = cfg.get("yolo_roi_w", 640)
            self.yolo_roi_h = cfg.get("yolo_roi_h", 480)

            # YOLO配置
            self.yolov4_cfg_path = get_vision_detector_dir() / cfg.get("yolov4_cfg_path", "./yolov4/cfg/iron_chip.cfg")
            self.yolov4_weights_path = get_vision_detector_dir() / cfg.get("yolov4_weights_path", "./yolov4/weights/iron_chip.weights")
            self.yolov4_names_path = get_vision_detector_dir() / cfg.get("yolov4_names_path", "./yolov4/names/obj.names")
            self.yolov4_conf_threshold = cfg.get("yolov4_conf_threshold", 0.5)
            self.yolov4_nms_threshold = cfg.get("yolov4_nms_threshold", 0.45)
            self.yolov4_input_w = cfg.get("yolov4_input_w", 608)
            self.yolov4_input_h = cfg.get("yolov4_input_h", 608)

            # 末端工具坐标
            self.tool_coord_x = cfg.get("tool_coord_x", 0)
            self.tool_coord_y = cfg.get("tool_coord_y", 0)
            self.tool_coord_z = cfg.get("tool_coord_z", 0)
            self.tool_coord_r = cfg.get("tool_coord_r", 0)

            # ===================== 下料算法配置加载 =====================
            self.unload_roi_x = cfg.get("unload_roi_x", 0)
            self.unload_roi_y = cfg.get("unload_roi_y", 0)
            self.unload_roi_w = cfg.get("unload_roi_w", 640)
            self.unload_roi_h = cfg.get("unload_roi_h", 480)
            self.unload_layer_count = cfg.get("unload_layer_count", 3)
            self.unload_layer_height = cfg.get("unload_layer_height", 50)
            self.unload_item_count_per_layer = cfg.get("unload_item_count_per_layer", 5)
            self.unload_item_interval = cfg.get("unload_item_interval", 80)
            self.unload_item_width = cfg.get("unload_item_width", 60)
            self.unload_depth_threshold = cfg.get("unload_depth_threshold", 20)

            self.config_loaded = True
            self._init_yolov4_model()

            return {"code": 0}
        except Exception as e:
            return {"code": -2, "err_msg": f"初始化失败: {str(e)}"}

    # ✅ 像素坐标+深度 → XYZ真实三维坐标 (针孔相机模型) 【无修改】
    def _pixel2world(self, pixel, depth_mm):
        x = (pixel[0] - self.camera_cx) * depth_mm / self.camera_fx
        y = (pixel[1] - self.camera_cy) * depth_mm / self.camera_fy
        z = depth_mm
        return (round(x, 2), round(y, 2), round(z, 2))

    # ✅ 【保留原函数】旋转矩形 顶边/底边 中点计算 【无修改】
    def _get_edge_center_point(self, rotated_rect):
        box = cv2.boxPoints(rotated_rect)
        box = np.int32(np.round(box))
        pts = box.tolist()
        pts_sorted_by_y = sorted(pts, key=lambda p: p[1])
        top_pts = pts_sorted_by_y[:2]
        bottom_pts = pts_sorted_by_y[-2:]

        if self.sort_rule == SortRule.SORT_BY_Y_ASC:
            cx = int((top_pts[0][0] + top_pts[1][0]) / 2)
            cy = int((top_pts[0][1] + top_pts[1][1]) / 2)
            return (cx, cy)
        elif self.sort_rule == SortRule.SORT_BY_Y_DESC:
            cx = int((bottom_pts[0][0] + bottom_pts[1][0]) / 2)
            cy = int((bottom_pts[0][1] + bottom_pts[1][1]) / 2)
            return (cx, cy)
        else:
            return (int(rotated_rect[0][0]), int(rotated_rect[0][1]))

    # 区域排序核心函数 【无修改】
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
        for i in range(len(regions)):
            regions[i]["region_id"] = 1  # 固定ID=1，因为永远只有1个结果

    # ==============================================================
    # ✅ ✅ ✅ 【直线检测算法完整版】_depth_segment_find_horizontal_line
    # 适配多ROI：新增feed_roi参数，默认使用上料ROI
    # ==============================================================
    def _depth_segment_find_horizontal_line(self, depth_img, use_feed_roi=True):
        regions = []
        h, w = depth_img.shape

        # 深度图预处理
        depth_filtered = cv2.medianBlur(depth_img, self.median_blur_kernel)
        depth_filtered = cv2.GaussianBlur(depth_filtered, (3, 3), self.gaussian_sigma)

        # 选择使用的ROI（上料ROI/其他ROI）
        if use_feed_roi:
            roi_x = self.feed_roi_x
            roi_y = self.feed_roi_y
            roi_w = self.feed_roi_w
            roi_h = self.feed_roi_h
        else:
            roi_x = self.unload_roi_x
            roi_y = self.unload_roi_y
            roi_w = self.unload_roi_w
            roi_h = self.unload_roi_h

        # 限定ROI区域
        roi_x_start = max(roi_x, 0)
        roi_x_end = min(roi_x + roi_w, w)
        roi_y_start = max(roi_y, 0)
        roi_y_end = min(roi_y + roi_h, h)

        # ✅ 提取ROI区域
        roi_depth = depth_filtered[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_h, roi_w = roi_depth.shape

        # ✅ 步骤1：创建深度边缘图（用于直线检测）
        # 将深度图转换为8位灰度图用于边缘检测
        depth_normalized = cv2.normalize(roi_depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        # 使用Canny边缘检测（深度跳变处产生边缘）
        edges = cv2.Canny(depth_normalized, 10, 30)

        # ✅ 步骤2：霍夫直线检测
        lines = cv2.HoughLinesP(edges,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=50,
                                minLineLength=roi_w * 0.3,  # 直线最小长度（ROI宽度的30%）
                                maxLineGap=30)

        # ✅ 步骤3：筛选水平直线
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # 计算直线角度（排除垂直线）
                if x2 != x1:  # 避免除以零
                    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi

                    # 筛选接近水平的直线（角度小于阈值）
                    if angle < 15:  # 小于15度视为水平
                        # 计算直线长度
                        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                        # 确保直线足够长
                        if length > roi_w * 0.1:  # 至少ROI宽度的20%
                            horizontal_lines.append({
                                'points': [(x1 + roi_x_start, y1 + roi_y_start),
                                           (x2 + roi_x_start, y2 + roi_y_start)],
                                'length': length,
                                'angle': angle,
                                'y_avg': (y1 + y2) / 2 + roi_y_start
                            })

        # ✅ 步骤4：根据排序规则选择目标直线
        target_line = None
        if horizontal_lines:
            # 按Y坐标排序
            if self.sort_rule == SortRule.SORT_BY_Y_ASC:
                # 从上到下：选择Y坐标最小的直线
                horizontal_lines.sort(key=lambda l: l['y_avg'])
            elif self.sort_rule == SortRule.SORT_BY_Y_DESC:
                # 从下到上：选择Y坐标最大的直线
                horizontal_lines.sort(key=lambda l: l['y_avg'], reverse=True)

            # 选择第一条符合条件的直线
            target_line = horizontal_lines[0]

        # ✅ 步骤5：从直线提取像素点并构造region
        if target_line:
            # 提取直线端点
            (x1, y1), (x2, y2) = target_line['points']

            # ✅ 计算直线的几何中点（修复点）
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            edge_center_point = (mid_x, mid_y)

            # ✅ 从直线上提取有效像素点
            target_pixels = []
            target_depth_sum = 0.0
            valid_pixel_count = 0

            # 生成直线上的采样点（使用Bresenham算法）
            line_points = self._bresenham_line(x1, y1, x2, y2)

            # 收集直线及其附近的有效像素点
            for (x, y) in line_points:
                # 检查是否在ROI和图像范围内
                if (roi_x_start <= x < roi_x_end and
                        roi_y_start <= y < roi_y_end):

                    curr_depth = depth_filtered[y, x]
                    # 深度有效性检查
                    if (curr_depth != self.depth_invalid and
                            self.feed_depth_min <= curr_depth <= self.feed_depth_max):
                        target_pixels.append((x, y))
                        target_depth_sum += curr_depth
                        valid_pixel_count += 1

            # 如果有效像素数足够
            if valid_pixel_count > 200:  # 与原代码一致的最小像素数
                area = valid_pixel_count
                avg_depth = int(target_depth_sum / area) if area > 0 else self.feed_depth_min

                # 计算像素中心点
                cx = int(np.mean([p[0] for p in target_pixels]))
                cy = int(np.mean([p[1] for p in target_pixels]))
                pixel_center = (cx, cy)

                # 构造旋转矩形
                pts = np.array(target_pixels, dtype=np.int32).reshape((-1, 1, 2))
                rotated_rect = cv2.minAreaRect(pts)
                rotate_angle = round(rotated_rect[2], 2)

                # 计算世界坐标
                world_xyz = self._pixel2world(edge_center_point, avg_depth)

                # ✅ 构造region结构（与原结构完全一致）
                regions.append({
                    "region_id": 1,
                    "pixel_center": pixel_center,
                    "edge_center_point": edge_center_point,  # ✅ 直线的几何中点
                    "world_xyz": world_xyz,
                    "rotate_angle": rotate_angle,
                    "rotated_rect": rotated_rect,
                    "area": area,
                    "avg_depth": avg_depth,
                    "color": (0, 0, 255)  # 固定红色，标识水平直线区域
                })

        # 排序（保留原逻辑）
        self._sort_regions(regions)
        self.detected_regions = regions
        return regions

    # ✅ 辅助函数：Bresenham直线算法（用于在两点间生成所有像素点）
    def _bresenham_line(self, x1, y1, x2, y2):
        """Bresenham直线算法，返回两点间所有整数坐标点"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # 确定步进方向
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        # 初始化误差
        err = dx - dy

        while True:
            points.append((x1, y1))

            # 到达终点
            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return points

    # ✅ 备选方案：使用更先进的LSD线段检测器
    def _depth_segment_find_horizontal_line_lsd(self, depth_img):
        """使用LSD（线段检测器）的版本"""
        regions = []
        h, w = depth_img.shape

        # 深度图预处理
        depth_filtered = cv2.medianBlur(depth_img, self.median_blur_kernel)
        depth_filtered = cv2.GaussianBlur(depth_filtered, (3, 3), self.gaussian_sigma)

        # 限定ROI区域（使用上料ROI）
        roi_x_start = max(self.feed_roi_x, 0)
        roi_x_end = min(self.feed_roi_x + self.feed_roi_w, w)
        roi_y_start = max(self.feed_roi_y, 0)
        roi_y_end = min(self.feed_roi_y + self.feed_roi_h, h)

        # ✅ 提取ROI区域
        roi_depth = depth_filtered[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # 转换为8位灰度图
        depth_normalized = cv2.normalize(roi_depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        # ✅ 使用LSD线段检测器
        lsd = cv2.createLineSegmentDetector()
        lines, width, prec, nfa = lsd.detect(depth_normalized)

        # ✅ 筛选水平线段
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.flatten()

                # 计算线段角度
                dx = x2 - x1
                dy = y2 - y1
                if dx != 0:
                    angle = np.arctan2(abs(dy), abs(dx)) * 180 / np.pi

                    # 筛选水平线段
                    if angle < 15:  # 水平阈值
                        length = np.sqrt(dx ** 2 + dy ** 2)

                        if length > roi_depth.shape[1] * 0.05:  # 足够长
                            horizontal_lines.append({
                                'points': [(x1 + roi_x_start, y1 + roi_y_start),
                                           (x2 + roi_x_start, y2 + roi_y_start)],
                                'length': length,
                                'angle': angle,
                                'y_avg': (y1 + y2) / 2 + roi_y_start
                            })

        # ✅ 步骤4：根据排序规则选择目标直线
        target_line = None
        if horizontal_lines:
            # 按Y坐标排序
            if self.sort_rule == SortRule.SORT_BY_Y_ASC:
                # 从上到下：选择Y坐标最小的直线
                horizontal_lines.sort(key=lambda l: l['y_avg'])
            elif self.sort_rule == SortRule.SORT_BY_Y_DESC:
                # 从下到上：选择Y坐标最大的直线
                horizontal_lines.sort(key=lambda l: l['y_avg'], reverse=True)

            # 选择第一条符合条件的直线
            target_line = horizontal_lines[0]

        # ✅ 步骤5：从直线提取像素点并构造region
        if target_line:
            # 提取直线端点
            (x1, y1), (x2, y2) = target_line['points']

            # ✅ 计算直线的几何中点（修复点）
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            edge_center_point = (mid_x, mid_y)

            # ✅ 从直线上提取有效像素点
            target_pixels = []
            target_depth_sum = 0.0
            valid_pixel_count = 0

            # 生成直线上的采样点（使用Bresenham算法）
            line_points = self._bresenham_line(x1, y1, x2, y2)

            # 收集直线及其附近的有效像素点
            for (x, y) in line_points:
                # 检查是否在ROI和图像范围内
                if (roi_x_start <= x < roi_x_end and
                        roi_y_start <= y < roi_y_end):

                    curr_depth = depth_filtered[y, x]
                    # 深度有效性检查
                    if (curr_depth != self.depth_invalid and
                            self.feed_depth_min <= curr_depth <= self.feed_depth_max):
                        target_pixels.append((x, y))
                        target_depth_sum += curr_depth
                        valid_pixel_count += 1

            # 如果有效像素数足够
            if valid_pixel_count > 200:  # 与原代码一致的最小像素数
                area = valid_pixel_count
                avg_depth = int(target_depth_sum / area) if area > 0 else self.feed_depth_min

                # 计算像素中心点
                cx = int(np.mean([p[0] for p in target_pixels]))
                cy = int(np.mean([p[1] for p in target_pixels]))
                pixel_center = (cx, cy)

                # 构造旋转矩形
                pts = np.array(target_pixels, dtype=np.int32).reshape((-1, 1, 2))
                rotated_rect = cv2.minAreaRect(pts)
                rotate_angle = round(rotated_rect[2], 2)

                # 计算世界坐标
                world_xyz = self._pixel2world(edge_center_point, avg_depth)

                # ✅ 构造region结构（与原结构完全一致）
                regions.append({
                    "region_id": 1,
                    "pixel_center": pixel_center,
                    "edge_center_point": edge_center_point,  # ✅ 直线的几何中点
                    "world_xyz": world_xyz,
                    "rotate_angle": rotate_angle,
                    "rotated_rect": rotated_rect,
                    "area": area,
                    "avg_depth": avg_depth,
                    "color": (0, 0, 255)  # 固定红色，标识水平直线区域
                })

        # 排序（保留原逻辑）
        self._sort_regions(regions)
        self.detected_regions = regions
        return regions

    # ===================== 【新增】物料缓存台检测算法 =====================
    def _material_check(self, depth_img):
        """物料缓存台检测：基于ROI内平均深度判断是否有物料"""
        exists_flag = DetectStatus.NOTHING
        coords = [0.0, 0.0, 0.0, 0.0]

        # 限定物料缓存台ROI范围
        h, w = depth_img.shape
        roi_x_start = max(self.material_roi_x, 0)
        roi_x_end = min(self.material_roi_x + self.material_roi_w, w)
        roi_y_start = max(self.material_roi_y, 0)
        roi_y_end = min(self.material_roi_y + self.material_roi_h, h)

        # 提取ROI深度图
        roi_depth = depth_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # 过滤无效深度值
        valid_depth_mask = (roi_depth != self.depth_invalid) & \
                           (roi_depth >= self.feed_depth_min) & \
                           (roi_depth <= self.feed_depth_max)
        valid_depth_values = roi_depth[valid_depth_mask]

        if len(valid_depth_values) > 0:
            # 计算ROI内有效深度的平均值
            avg_depth = np.mean(valid_depth_values)
            print(f"material depth = {avg_depth}")

            # 判断是否在物料深度范围内
            if self.material_depth_min <= avg_depth <= self.material_depth_max:
                exists_flag = DetectStatus.EXIST

                # 计算ROI中心像素坐标
                roi_center_x = roi_x_start + (roi_x_end - roi_x_start) // 2
                roi_center_y = roi_y_start + (roi_y_end - roi_y_start) // 2

                # 转换为世界坐标
                world_xyz = self._pixel2world((roi_center_x, roi_center_y), avg_depth)
                x, y, z = world_xyz
                r = 0.0  # 旋转角度默认0

                coords = [x, y, z, r]

        return exists_flag, coords

    # ===================== 【新增】下料算法 =====================
    def _unload_check(self, depth_img):
        """下料算法：识别可放置长条形产品的区域坐标"""
        exists_flag = DetectStatus.NOTHING
        coords = [0.0, 0.0, 0.0, 0.0]

        # 1. 预处理深度图
        depth_filtered = cv2.medianBlur(depth_img, self.median_blur_kernel)
        depth_filtered = cv2.GaussianBlur(depth_filtered, (3, 3), self.gaussian_sigma)

        # 2. 限定下料ROI范围
        h, w = depth_img.shape
        roi_x_start = max(self.unload_roi_x, 0)
        roi_x_end = min(self.unload_roi_x + self.unload_roi_w, w)
        roi_y_start = max(self.unload_roi_y, 0)
        roi_y_end = min(self.unload_roi_y + self.unload_roi_h, h)

        # 3. 生成每层的理论坐标
        layer_coords = []
        # 计算每层的基准深度（从底层开始）
        base_depth = self.feed_depth_min  # 底层基准深度
        for layer_idx in range(self.unload_layer_count):
            # 每层的深度偏移
            layer_depth = base_depth + layer_idx * self.unload_layer_height

            # 计算每层第一个产品的X坐标（居中排列）
            total_width = self.unload_item_count_per_layer * self.unload_item_width + \
                          (self.unload_item_count_per_layer - 1) * self.unload_item_interval
            start_x = roi_x_start + (roi_x_end - roi_x_start - total_width) // 2
            start_y = roi_y_start + (roi_y_end - roi_y_start) // 2  # 层Y坐标居中

            # 生成该层所有产品的坐标
            for item_idx in range(self.unload_item_count_per_layer):
                item_x = start_x + item_idx * (
                            self.unload_item_width + self.unload_item_interval) + self.unload_item_width // 2
                item_y = start_y
                layer_coords.append({
                    "layer_idx": layer_idx,
                    "item_idx": item_idx,
                    "pixel_x": item_x,
                    "pixel_y": item_y,
                    "target_depth": layer_depth,
                    "is_empty": False
                })

        # 4. 检测每个位置是否为空（深度差超过阈值则视为空）
        empty_positions = []
        for pos in layer_coords:
            px = int(pos["pixel_x"])
            py = int(pos["pixel_y"])

            # 检查像素坐标是否在有效范围内
            if roi_x_start <= px < roi_x_end and roi_y_start <= py < roi_y_end:
                curr_depth = depth_filtered[py, px]

                # 判断是否为空：无效深度 或 深度差超过阈值
                if (curr_depth == self.depth_invalid) or \
                        (abs(curr_depth - pos["target_depth"]) > self.unload_depth_threshold):
                    pos["is_empty"] = True
                    empty_positions.append(pos)

        # 5. 找到第一个空余位置（优先低层，同层优先左侧）
        if empty_positions:
            # 按层号升序、同层按物品索引升序排序
            empty_positions.sort(key=lambda p: (p["layer_idx"], p["item_idx"]))
            first_empty = empty_positions[0]

            exists_flag = DetectStatus.EXIST

            # 计算该位置的世界坐标
            pixel_x = first_empty["pixel_x"]
            pixel_y = first_empty["pixel_y"]
            target_depth = first_empty["target_depth"]

            world_xyz = self._pixel2world((pixel_x, pixel_y), target_depth)
            x, y, z = world_xyz
            r = 0.0  # 旋转角度默认0

            # 叠加工具坐标偏移
            x += self.tool_coord_x
            y += self.tool_coord_y
            z += self.tool_coord_z
            r += self.tool_coord_r

            coords = [x, y, z, r]

        return exists_flag, coords

    # ===================== 【对外接口2 - 检测接口】核心 =====================
    def detect(self, ptype, rgb_img, depth_img):
        try:
            if not self.config_loaded:
                return {"code": -1, "err_msg": "请先调用初始化函数加载配置"}
            if depth_img is None or depth_img.dtype != np.uint16:
                return {"code": -2, "err_msg": "深度图格式错误，必须是CV_16UC1单通道格式"}
            if ptype < 1 or ptype > 4:
                return {"code": -3, "err_msg": "ptype类型错误，仅支持1/2/3/4"}

            coords = [0.0, 0.0, 0.0, 0.0]
            exists_flag = DetectStatus.UNKNOWN
            regions = []

            # 分支处理不同检测类型
            if ptype == PType.MATERIAL_CHECK:  # 物料缓存台
                exists_flag, coords = self._material_check(depth_img)
            elif ptype == PType.IRON_CHIP_CHECK:  # 铁屑
                exists_flag, coords = self._judge_detect_result(regions, ptype, rgb_img)
            elif ptype == PType.FEED_CHECK:  # 上料
                regions = self._depth_segment_find_horizontal_line(depth_img, use_feed_roi=True)
                if not regions:
                    exists_flag = DetectStatus.NOTHING
                else:
                    exists_flag = DetectStatus.EXIST
                    # 提取第一个区域的坐标
                    main_region = regions[0]
                    x, y, z = main_region["world_xyz"]
                    r = main_region["rotate_angle"]
                    x += self.tool_coord_x
                    y += self.tool_coord_y
                    z += self.tool_coord_z
                    r += self.tool_coord_r
                    coords = [x, y, z, r]
            elif ptype == PType.UNLOAD_CHECK:  # 下料
                exists_flag, coords = self._unload_check(depth_img)

            return {
                "code": 0,
                "result": {
                    "ptype": ptype,
                    "coords": coords,
                    "exists": exists_flag
                },
                "err_msg": ""
            }
        except Exception as e:
            return {"code": -99, "err_msg": f"检测异常: {str(e)}"}

    # 检测逻辑判断
    def _judge_detect_result(self, regions, ptype, rgb_img):
        coords = [0.0, 0.0, 0.0, 0.0]
        exists_flag = DetectStatus.UNKNOWN

        if ptype == PType.IRON_CHIP_CHECK:
            has_chip, chip_boxes = self._yolov4_detect_chip(rgb_img)
            if has_chip:
                exists_flag = DetectStatus.EXIST
            else:
                exists_flag = DetectStatus.NOTHING
            return exists_flag, coords

        # 原有逻辑 完全不变
        if ptype == PType.MATERIAL_CHECK or ptype == PType.FEED_CHECK:
            if not regions:
                exists_flag = DetectStatus.NOTHING
                return exists_flag, coords
            else:
                exists_flag = DetectStatus.EXIST

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

        return exists_flag, coords

    # ===================== 绘制函数 - 可视化水平直线 + 原标注 =====================
    def draw_result(self, rgb, detect_res):
        draw_img = rgb.copy()
        if detect_res["code"] != 0 or not detect_res["result"]:
            cv2.putText(draw_img, "DETECT ERR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return draw_img

        res = detect_res["result"]
        ptype = res["ptype"]
        coords = res["coords"]
        exists_flag = res["exists"]
        x, y, z, r = coords

        status_text = "EXIST" if exists_flag == DetectStatus.EXIST else "NOTHING"
        status_color = (0, 255, 0) if exists_flag == DetectStatus.EXIST else (0, 0, 255)
        cv2.putText(draw_img, f"STATUS: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # 检测类型
        type_dict = {1: "Material", 2: "Feed", 3: "Unload", 4: "Iron"}
        cv2.putText(draw_img, f"TYPE: {type_dict[ptype]}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 绘制坐标信息
        coord_text = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f} R:{r:.1f}"
        cv2.putText(draw_img, coord_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 绘制对应ROI框
        if ptype == PType.MATERIAL_CHECK:
            # 物料缓存台ROI（绿色）
            cv2.rectangle(draw_img, (self.material_roi_x, self.material_roi_y),
                          (self.material_roi_x + self.material_roi_w, self.material_roi_y + self.material_roi_h),
                          (0, 255, 0), 2)
            cv2.putText(draw_img, "Material ROI", (self.material_roi_x + 5, self.material_roi_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif ptype == PType.FEED_CHECK:
            # 上料ROI（蓝色）
            cv2.rectangle(draw_img, (self.feed_roi_x, self.feed_roi_y),
                          (self.feed_roi_x + self.feed_roi_w, self.feed_roi_y + self.feed_roi_h),
                          (255, 0, 0), 2)
            cv2.putText(draw_img, "Feed ROI", (self.feed_roi_x + 5, self.feed_roi_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        elif ptype == PType.UNLOAD_CHECK:
            # 下料ROI（黄色）
            cv2.rectangle(draw_img, (self.unload_roi_x, self.unload_roi_y),
                          (self.unload_roi_x + self.unload_roi_w, self.unload_roi_y + self.unload_roi_h),
                          (0, 255, 255), 2)
            cv2.putText(draw_img, "Unload ROI", (self.unload_roi_x + 5, self.unload_roi_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif ptype == PType.IRON_CHIP_CHECK:
            # YOLO检测ROI（紫色）
            cv2.rectangle(draw_img, (self.yolo_roi_x, self.yolo_roi_y),
                          (self.yolo_roi_x + self.yolo_roi_w, self.yolo_roi_y + self.yolo_roi_h),
                          (255, 0, 255), 2)
            cv2.putText(draw_img, "YOLO ROI", (self.yolo_roi_x + 5, self.yolo_roi_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            # 绘制YOLO铁屑检测框
            for (x1, y1, w, h, conf) in self.detect_iron_chips:
                cv2.rectangle(draw_img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                cv2.putText(draw_img, f"{conf}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
        exists_flag = res["exists"]
        x, y, z, r = coords

        # 1. 绘制原有所有信息：状态、类型、坐标
        status_text = "EXIST" if exists_flag == DetectStatus.EXIST else "NOTHING"
        status_color = (0, 255, 0) if exists_flag == DetectStatus.EXIST else (0, 0, 255)
        cv2.putText(draw_img, f"STATUS: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        # 检测类型
        type_dict = {1: "Material", 2: "Feed", 3: "Unload", 4: "Iron"}
        cv2.putText(draw_img, f"TYPE: {type_dict[ptype]}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        coord_text = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f} R:{r:.1f}"
        cv2.putText(draw_img, coord_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 2. 绘制对应ROI框
        roi_info = {
            PType.MATERIAL_CHECK: ("Material ROI", (self.material_roi_x, self.material_roi_y),
                                   (self.material_roi_x + self.material_roi_w,
                                    self.material_roi_y + self.material_roi_h), (0, 255, 0)),
            PType.FEED_CHECK: ("Feed ROI", (self.feed_roi_x, self.feed_roi_y),
                               (self.feed_roi_x + self.feed_roi_w, self.feed_roi_y + self.feed_roi_h), (255, 0, 0)),
            PType.UNLOAD_CHECK: ("Unload ROI", (self.unload_roi_x, self.unload_roi_y),
                                 (self.unload_roi_x + self.unload_roi_w, self.unload_roi_y + self.unload_roi_h),
                                 (0, 255, 255)),
            PType.IRON_CHIP_CHECK: ("YOLO ROI", (self.yolo_roi_x, self.yolo_roi_y),
                                    (self.yolo_roi_x + self.yolo_roi_w, self.yolo_roi_y + self.yolo_roi_h),
                                    (255, 0, 255))
        }
        if ptype in roi_info:
            roi_name, roi_start, roi_end, roi_color = roi_info[ptype]
            # 绘制ROI框
            cv2.rectangle(draw_img, roi_start, roi_end, roi_color, 2)
            cv2.putText(draw_img, roi_name, (roi_start[0] + 5, roi_start[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

        # 3. 绘制旋转矩形框（仅上料检测）
        if ptype == PType.FEED_CHECK and len(self.detected_regions) > 0:
            region = self.detected_regions[0]
            rotated_rect = region["rotated_rect"]
            box = cv2.boxPoints(rotated_rect)
            box = np.int32(box)
            cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 2)

            # 绘制中心点
            cx, cy = region["pixel_center"]
            cv2.circle(draw_img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(draw_img, f"Center ({cx},{cy})", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 4. 绘制YOLO铁屑检测框（仅铁屑检测）
        if ptype == PType.IRON_CHIP_CHECK and len(self.detect_iron_chips) > 0:
            for (x1, y1, w, h, conf) in self.detect_iron_chips:
                cv2.rectangle(draw_img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                cv2.putText(draw_img, f"Chip {conf}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 5. 绘制下料区域的理论放置位置（仅下料检测）
        if ptype == PType.UNLOAD_CHECK:
            h, w = draw_img.shape[:2]
            roi_x_start = max(self.unload_roi_x, 0)
            roi_x_end = min(self.unload_roi_x + self.unload_roi_w, w)
            roi_y_start = max(self.unload_roi_y, 0)

            # 计算每层的理论坐标
            for layer_idx in range(self.unload_layer_count):
                total_width = self.unload_item_count_per_layer * self.unload_item_width + \
                              (self.unload_item_count_per_layer - 1) * self.unload_item_interval
                start_x = roi_x_start + (roi_x_end - roi_x_start - total_width) // 2
                start_y = roi_y_start + (self.unload_roi_h // 2) + layer_idx * 20  # 分层显示

                # 绘制该层所有产品位置
                for item_idx in range(self.unload_item_count_per_layer):
                    item_x = start_x + item_idx * (
                                self.unload_item_width + self.unload_item_interval) + self.unload_item_width // 2
                    item_y = start_y

                    # 绘制位置框
                    cv2.rectangle(draw_img,
                                  (item_x - self.unload_item_width // 2, item_y - 10),
                                  (item_x + self.unload_item_width // 2, item_y + 10),
                                  (255, 255, 0), 1)
                    # 标注层号和物品索引
                    cv2.putText(draw_img, f"L{layer_idx}I{item_idx}", (item_x - 20, item_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # 绘制第一个空余位置（红色高亮）
            if exists_flag == DetectStatus.EXIST:
                target_x = int(coords[0] * self.camera_fx / coords[2] + self.camera_cx)
                target_y = int(coords[1] * self.camera_fy / coords[2] + self.camera_cy)
                cv2.circle(draw_img, (target_x, target_y), 8, (0, 0, 255), -1)
                cv2.putText(draw_img, "TARGET", (target_x + 10, target_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return draw_img

    def depth_pseudo_color(self, depth_img):
        # 1. 深度图转【伪彩色热力图】核心步骤 (深度值归一化到0-255 + 上色)
        depth_show = depth_img.copy()
        # 归一化深度值到 0-255 (只对ROI内的有效深度做归一化，排除无效值)
        # roi_x_start = max(self.roi_x, 0)
        # roi_x_end = min(self.roi_x + self.roi_w, depth_show.shape[1])
        # roi_y_start = max(self.roi_y, 0)
        # roi_y_end = min(self.roi_y + self.roi_h, depth_show.shape[0])
        # roi_depth = depth_show[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_depth = depth_show
        # 过滤无效深度值，只计算有效深度的最大最小值
        valid_depth = roi_depth[
            (roi_depth != self.depth_invalid) & (roi_depth >= self.feed_depth_min) & (roi_depth <= self.feed_depth_max)]
        if len(valid_depth) > 0:
            depth_min = valid_depth.min()
            depth_max = valid_depth.max()
        else:
            depth_min = self.feed_depth_min
            depth_max = self.feed_depth_max
        # 归一化+转uint8
        depth_show = np.clip(depth_show, depth_min, depth_max)
        depth_show = ((depth_show - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        # 深度图转伪彩色 (COLORMAP_JET 工业视觉最常用，近蓝远红，可换其他)
        depth_pseudo_color = cv2.applyColorMap(depth_show, cv2.COLORMAP_JET)

        return depth_pseudo_color


# ===================== 测试主函数 调用示例 =====================
if __name__ == "__main__":
    detector = RGBDDetector()
    product_no = "M001"
    init_res = detector.init(product_no)
    if init_res["code"] != 0:
        print(f"初始化失败: {init_res['err_msg']}")
        exit(-1)
    print("初始化成功！")

    rgb_img = cv2.imread("./rgb_image.png")
    depth_img = cv2.imread("./depth_1768713398274.png", cv2.IMREAD_UNCHANGED)

    if rgb_img is None or depth_img is None:
        print("读取图像失败，请检查路径！")
        exit(-1)

    if depth_img.dtype == np.uint8:
        depth_img = depth_img.astype(np.uint16) * 20
    elif depth_img.dtype == np.uint16:
        depth_img8u = depth_img.astype(np.uint8)
        cv2.imwrite("./depth_8u.png", depth_img8u)

    # 切换排序规则 → 联动扫描方向
    # detector.sort_rule = SortRule.SORT_BY_Y_DESC    # Y降序 → 从底部向上找水平直线
    # detector.sort_rule = SortRule.SORT_BY_Y_ASC   # Y升序 → 从顶部向下找水平直线

    # ptype = PType.MATERIAL_CHECK
    # ptype = PType.FEED_CHECK
    ptype = PType.UNLOAD_CHECK
    detect_res = detector.detect(ptype, rgb_img, depth_img)
    print("检测结果:\n", json.dumps(detect_res, ensure_ascii=False, indent=2))

    depth_color = detector.depth_pseudo_color(depth_img)

    result_img = detector.draw_result_with_rotated_box(depth_color, detect_res)
    cv2.imshow("result-line", result_img)
    cv2.imwrite("./detect_result_horizontal_line.jpg", result_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
