import traceback

import cv2
import numpy as np
import json
import os
import random
import re
from pathlib import Path
from src.utils.path_helper import get_vision_detector_dir
from src.utils import logger

# ===================== 常量定义 =====================
class SortRule:
    SORT_BY_Y_DESC = 0    # 按Y降序 → 从ROI底部向上扫描，找第一个水平直线 ✅核心联动
    SORT_BY_Y_ASC = 1     # 按Y升序 → 从ROI顶部向下扫描，找第一个水平直线 ✅核心联动
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
    FEED_END_CHECK = 5

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

        self.debug_mode = False

        # =====================                        
        # 末端工具相对相机中心的坐标
        self.tool_coord_x = 0
        self.tool_coord_y = 0
        self.tool_coord_z = 0
        self.tool_coord_r = 0

        # 产品尺寸参数
        self.product_height = 100
        self.product_width = 120
        self.interval_height = 20
        self.feed_depth_start = 340
        # self.tray_start_height = 1000
        self.product_count_per_layer = 8

        # ===================== 
        # 端面水平线检测配置（相机翻转90度，检测产品端面）
        # 端面检测ROI
        self.end_roi_x = 0
        self.end_roi_y = 0
        self.end_roi_w = 640
        self.end_roi_h = 480
        self.end_depth_start = 340         # 顶层产品初始高度
        
        # 端面检测参数
        self.end_depth_min = 400
        self.end_depth_max = 450
        self.end_depth_thresh = 10          # Canny阈值
        self.end_min_length = 10           # 最小线段长度（像素）- 端面边缘较短
        self.end_edge_index = 0             # 返回第几个边缘
        self.end_offset_x = 0               # X方向抓取偏移
        self.end_offset_y = 0               # Y方向抓取偏移
        self.end_offset_z = 0               # Z方向抓取偏移
        


        # ===================== 
        # 1. 上料ROI（原ROI）
        self.feed_roi_x = 0
        self.feed_roi_y = 0
        self.feed_roi_w = 640
        self.feed_roi_h = 480

        
        # 上料深度区间过滤参数
        self.feed_depth_min = 50    
        self.feed_depth_max = 2000
        self.feed_depth_thresh = 10
        self.feed_min_length = 200
        self.feed_edge_index = 0
        self.feed_offset_x = 0
        self.feed_offset_y = 0
        self.feed_offset_z = 0

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
        self.yolov4_cfg_path = "./yolov4/iron_chip.cfg"    # yolov4配置文件路径
        self.yolov4_weights_path = "./yolov4/iron_chip.weights" # yolov4权重文件路径
        self.yolov4_names_path = "./yolov4/iron_chip.names"        # 类别名称文件路径
        self.yolov4_conf_threshold = 0.5                           
        self.yolov4_nms_threshold = 0.45                           
        self.yolov4_input_w = 608                                  # 输入宽度
        self.yolov4_input_h = 608                                  # 输入高度
        self.yolov4_net = None
        self.yolov4_classes = []
        self.detect_iron_chips = []

        # ===================== 
        # 下料算法配置参数
        self.unload_roi_x = 0               # 下料区域ROI X
        self.unload_roi_y = 0               # 下料区域ROI Y
        self.unload_roi_w = 640             # 下料区域ROI 宽度
        self.unload_roi_h = 480             # 下料区域ROI 高度
        self.unload_layer_count = 3         # 下料层数
        self.unload_layer_height = 50       # 层高（mm）
        self.unload_item_count_per_layer = 5# 每层放置产品数量
        self.unload_item_interval = 80      # 产品间隔（像素/物理尺寸，根据标定转换）
        self.unload_item_width = 600        # 产品宽度（像素）
        self.unload_item_height = 60        # 产品高度（像素）
        self.unload_depth_threshold = 20    # 深度差阈值：判断该位置是否有物料
        self.unload_depth_min = 100         # 物料存在的最小深度
        self.unload_depth_max = 500         # 物料存在的最大深度
        
        # =====================   
        random.seed(10)
        # 保存分割后的区域结果，用于绘图接口调用
        self.detected_regions = []


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
        blob = cv2.dnn.blobFromImage(rgb_img, 1 / 255.0, (self.yolov4_input_w, self.yolov4_input_h), swapRB=True, crop=False)
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
                    self.detect_iron_chips.append( (x1, y1, w, h, round(conf, 2)) )
        
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
                #cfg = json.load(f)
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

            # 排序
            self.sort_rule = cfg.get("sort_rule", 0)

            # 图像预处理参数
            self.depth_invalid = cfg.get("depth_invalid", 0)
            self.median_blur_kernel = cfg.get("median_blur_kernel", 3)
            self.gaussian_sigma = cfg.get("gaussian_sigma", 1.2)
                       
            # 末端工具坐标
            self.tool_coord_x = cfg.get("tool_coord_x", 0)
            self.tool_coord_y = cfg.get("tool_coord_y", 0)
            self.tool_coord_z = cfg.get("tool_coord_z", 0)
            self.tool_coord_r = cfg.get("tool_coord_r", 0)

            # 产品参数
            self.product_height = cfg.get("product_height", 100)
            self.product_width = cfg.get("product_width", 160)
            self.interval_height = cfg.get("interval_height", 20)
            # self.tray_start_height = cfg.get("tray_start_height", 1000)
            self.product_count_per_layer = cfg.get("product_count_per_layer", 8)
           
            # ===================== 
            # 端面水平线检测配置加载
            self.end_roi_x = cfg.get("end_roi_x", 0)
            self.end_roi_y = cfg.get("end_roi_y", 0)
            self.end_roi_w = cfg.get("end_roi_w", 640)
            self.end_roi_h = cfg.get("end_roi_h", 480)
            
            self.end_depth_min = cfg.get("end_depth_min", 400)
            self.end_depth_max = cfg.get("end_depth_max", 450)
            self.end_depth_thresh = cfg.get("end_depth_thresh", 10)
            self.end_min_length = cfg.get("end_min_length", 10)
            self.end_edge_index = cfg.get("end_edge_index", 0)
            self.end_offset_x = cfg.get("end_offset_x", 0)
            self.end_offset_y = cfg.get("end_offset_y", 0)
            self.end_offset_z = cfg.get("end_offset_z", 0)
            
            self.end_product_height = cfg.get("end_product_height", 100)
            self.end_product_width = cfg.get("end_product_width", 120)
            self.end_interval_height = cfg.get("end_interval_height", 20)
            # self.end_depth_start = cfg.get("end_depth_start", 340)

            # ===================== 
            # 1. 上料ROI
            self.feed_roi_x = cfg.get("feed_roi_x", 0)
            self.feed_roi_y = cfg.get("feed_roi_y", 0)
            self.feed_roi_w = cfg.get("feed_roi_w", 640)
            self.feed_roi_h = cfg.get("feed_roi_h", 480)

            # 上料算法参数
            self.feed_depth_min = cfg.get("feed_depth_min", 50)
            self.feed_depth_max = cfg.get("feed_depth_max", 2000)
            # self.feed_depth_start = cfg.get("feed_depth_start", 340)
            self.feed_depth_thresh = cfg.get("feed_depth_thresh", 10)
            self.feed_min_length = cfg.get("feed_min_length", 300)
            self.feed_edge_index = cfg.get("feed_edge_index", 0)
            self.feed_offset_x = cfg.get("feed_offset_x", 0)
            self.feed_offset_y = cfg.get("feed_offset_y", 0)
            self.feed_offset_z = cfg.get("feed_offset_z", 0)
            
            # 模板匹配算法，备用
            self.template_type = cfg.get("template_type", "")
            self.template_path = cfg.get("template_path", "")
            self.template_threshold = cfg.get("template_threshold", 0.6)
            self.template_edge_offset = cfg.get("template_edge_offset", 0)
            
            # ===================== 
            # 2. 物料缓存台ROI + 深度范围
            self.material_roi_x = cfg.get("material_roi_x", 0)
            self.material_roi_y = cfg.get("material_roi_y", 0)
            self.material_roi_w = cfg.get("material_roi_w", 640)
            self.material_roi_h = cfg.get("material_roi_h", 480)
            self.material_depth_min = cfg.get("material_depth_min", 100)
            self.material_depth_max = cfg.get("material_depth_max", 500)

            # ===================== 
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
 

            # ===================== 
            # 下料算法配置加载
            self.unload_roi_x = cfg.get("unload_roi_x", 0)
            self.unload_roi_y = cfg.get("unload_roi_y", 0)
            self.unload_roi_w = cfg.get("unload_roi_w", 640)
            self.unload_roi_h = cfg.get("unload_roi_h", 480)
            
            # 下料算法参数
            self.unload_layer_count = cfg.get("unload_layer_count", 3)
            self.unload_layer_height = cfg.get("unload_layer_height", 50)
            self.unload_item_count_per_layer = cfg.get("unload_item_count_per_layer", 5)
            self.unload_item_interval = cfg.get("unload_item_interval", 80)
            self.unload_item_width = cfg.get("unload_item_width", 600)
            self.unload_item_height = cfg.get("unload_item_height", 60)
            self.unload_depth_threshold = cfg.get("unload_depth_threshold", 20)

            # ===================== 
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
        return (round(x,2), round(y,2), round(z,2))

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
    def _sort_regions(self, regions, sort_rule):
        if not regions:
            return
        if sort_rule == SortRule.SORT_BY_Y_DESC:
            regions.sort(key=lambda x: x["pixel_center"][1], reverse=True)
        elif sort_rule == SortRule.SORT_BY_Y_ASC:
            regions.sort(key=lambda x: x["pixel_center"][1])
        elif sort_rule == SortRule.SORT_BY_X_DESC:
            regions.sort(key=lambda x: x["pixel_center"][0], reverse=True)
        elif sort_rule == SortRule.SORT_BY_X_ASC:
            regions.sort(key=lambda x: x["pixel_center"][0])
        elif sort_rule == SortRule.SORT_BY_AREA_DESC:
            regions.sort(key=lambda x: x["area"], reverse=True)
        elif sort_rule == SortRule.SORT_BY_DEPTH_ASC:
            regions.sort(key=lambda x: x["avg_depth"])
        for i in range(len(regions)):
            regions[i]["region_id"] = i + 1 # 固定ID=1，因为永远只有1个结果

    # 线段合并函数
    def _merge_nearby_lines(self, lines, max_gap=20, angle_threshold=10):
        """
        合并接近水平且端点距离较近的线段
        
        参数:
            lines: 线段列表，每个元素为字典，包含:
                - 'points': [(x1,y1), (x2,y2)] 线段端点
                - 'length': 线段长度
                - 'angle': 线段角度(度)
                - 'y_avg': 线段平均Y坐标
            max_gap: 最大端点距离阈值(像素)
            angle_threshold: 角度差异阈值(度)，小于此值认为方向相近
            
        返回:
            合并后的线段列表
        """
        if not lines:
            return []
        
        # 按Y坐标排序，便于从上到下或从下到上处理
        lines_sorted = sorted(lines, key=lambda l: l['y_avg'])
        
        merged_lines = []
        used_indices = set()
        
        for i in range(len(lines_sorted)):
            if i in used_indices:
                continue
                
            current_line = lines_sorted[i]
            (x1_curr, y1_curr), (x2_curr, y2_curr) = current_line['points']
            
            # 确定线段的左右端点（保证x1 <= x2）
            if x1_curr > x2_curr:
                x1_curr, x2_curr = x2_curr, x1_curr
                y1_curr, y2_curr = y2_curr, y1_curr
            
            merged_line = {
                'points': [(x1_curr, y1_curr), (x2_curr, y2_curr)],
                'length': current_line['length'],
                'angle': current_line['angle'],
                'y_avg': current_line['y_avg']
            }
            
            used_indices.add(i)
            
            # 寻找与当前线段可以合并的线段
            for j in range(i+1, len(lines_sorted)):
                if j in used_indices:
                    continue
                    
                other_line = lines_sorted[j]
                (x1_other, y1_other), (x2_other, y2_other) = other_line['points']
                
                # 确定其他线段的左右端点
                if x1_other > x2_other:
                    x1_other, x2_other = x2_other, x1_other
                    y1_other, y2_other = y2_other, y1_other
                
                # 检查角度是否相近（水平方向）
                angle_diff = abs(current_line['angle'] - other_line['angle'])
                if angle_diff > angle_threshold:
                    continue
                
                # 检查Y坐标是否相近（水平线应该在同一水平线上）
                y_diff = abs(current_line['y_avg'] - other_line['y_avg'])
                if y_diff > max_gap:  # Y方向阈值严格一些
                    continue
                
                # 检查端点距离
                # 计算当前线段与其他线段四个端点的最小距离
                # distances = [
                #     self._point_to_segment_distance((x1_other, y1_other), merged_line['points']),
                #     self._point_to_segment_distance((x2_other, y2_other), merged_line['points'])
                # ]
                # min_distance = min(distances)
                
                # 如果最小距离在阈值内，则合并线段
                if True: # min_distance <= max_gap * 10:
                    # 合并线段：取所有端点的最小X和最大X
                    all_x = [x1_curr, x2_curr, x1_other, x2_other]
                    all_y = [y1_curr, y2_curr, y1_other, y2_other]
                    
                    # 由于是水平线，我们使用加权平均计算Y坐标
                    # 根据长度加权
                    weight_curr = current_line['length']
                    weight_other = other_line['length']
                    merged_y = (y1_curr * weight_curr + y1_other * weight_other) / (weight_curr + weight_other)
                    
                    # 计算新的端点
                    new_x1 = min(all_x)
                    new_x2 = max(all_x)
                    new_y1 = merged_y
                    new_y2 = merged_y
                    
                    # 计算新的线段属性
                    new_length = new_x2 - new_x1
                    new_angle = 0  # 因为是水平线
                    new_y_avg = merged_y
                    
                    # 更新合并后的线段
                    merged_line['points'] = [(new_x1, new_y1), (new_x2, new_y2)]
                    merged_line['length'] = new_length
                    merged_line['angle'] = new_angle
                    merged_line['y_avg'] = new_y_avg
                    
                    used_indices.add(j)
            
            merged_lines.append(merged_line)
        
        return merged_lines

    def _calculate_line_distance(self, line1, line2):
        """
        计算两条线段延长线之间的最近距离
        
        参数:
            line1, line2: 线段字典，包含'points'键
            
        返回:
            两条线段延长线之间的最近距离
        """
        (x1_1, y1_1), (x2_1, y2_1) = line1['points']
        (x1_2, y1_2), (x2_2, y2_2) = line2['points']
        
        # 确定每条线段的左右端点
        if x1_1 > x2_1:
            x1_1, x2_1 = x2_1, x1_1
            y1_1, y2_1 = y2_1, y1_1
        
        if x1_2 > x2_2:
            x1_2, x2_2 = x2_2, x1_2
            y1_2, y2_2 = y2_2, y1_2
        
        # 计算两条直线的方程
        # 线段1的方程
        if x2_1 != x1_1:
            k1 = (y2_1 - y1_1) / (x2_1 - x1_1)
            b1 = y1_1 - k1 * x1_1
        else:
            # 垂直线
            k1 = None
            x_const1 = x1_1
        
        # 线段2的方程
        if x2_2 != x1_2:
            k2 = (y2_2 - y1_2) / (x2_2 - x1_2)
            b2 = y1_2 - k2 * x1_2
        else:
            # 垂直线
            k2 = None
            x_const2 = x1_2
        
        # 计算两条直线之间的平均距离
        if k1 is not None and k2 is not None:
            # 两条都是斜线
            # 计算在X重叠区域的平均距离
            overlap_x_start = max(x1_1, x1_2)
            overlap_x_end = min(x2_1, x2_2)
            
            if overlap_x_end > overlap_x_start:
                # 有重叠区域，在重叠区域采样计算平均距离
                sample_points = np.linspace(overlap_x_start, overlap_x_end, num=20)
                distances = []
                for x in sample_points:
                    y1 = k1 * x + b1
                    y2 = k2 * x + b2
                    distances.append(abs(y1 - y2))
                
                return np.mean(distances)
            else:
                # 没有重叠区域，计算两条直线之间的最短距离
                # 平行线之间的距离公式：|b2 - b1| / sqrt(k^2 + 1)
                if abs(k1 - k2) < 1e-6:  # 平行
                    distance = abs(b2 - b1) / np.sqrt(k1**2 + 1)
                    return distance
                else:
                    # 不平行的直线，计算交点处距离为0
                    return 0.0
        
        elif k1 is None and k2 is not None:
            # 线段1是垂直线，线段2是斜线
            # 计算垂直线与斜线交点的Y坐标
            y_on_line2 = k2 * x_const1 + b2
            # 取垂直线中点的Y坐标
            y_mid_line1 = (y1_1 + y2_1) / 2
            return abs(y_on_line2 - y_mid_line1)
        
        elif k1 is not None and k2 is None:
            # 线段1是斜线，线段2是垂直线
            y_on_line1 = k1 * x_const2 + b1
            y_mid_line2 = (y1_2 + y2_2) / 2
            return abs(y_on_line1 - y_mid_line2)
        
        else:
            # 两条都是垂直线
            return abs(x_const1 - x_const2)
    
    # 基于延长线的合并函数：
    def _merge_lines_by_extension(self, lines, end_threshold = 100, max_extension_gap=20, angle_threshold=10, y_threshold=10):
        """
        基于延长线距离合并线段
        
        参数:
            lines: 线段列表，每个元素为字典，包含:
                - 'points': [(x1,y1), (x2,y2)] 线段端点
                - 'length': 线段长度
                - 'angle': 线段角度(度)
                - 'y_avg': 线段平均Y坐标
            end_threshold: 端点之间的最近距离的阈值
            max_extension_gap: 延长线之间的最大允许距离(像素)
            angle_threshold: 角度差异阈值(度)
            y_threshold: Y坐标差异阈值(像素)，用于水平线
            
        返回:
            合并后的线段列表
        """
        if not lines:
            return []
        
        # 按X坐标排序（对于水平线）
        lines_sorted = sorted(lines, key=lambda l: min(l['points'][0][0], l['points'][1][0]))
        
        merged_lines = []
        used_indices = set()
        
        for i in range(len(lines_sorted)):
            if i in used_indices:
                continue
                
            current_line = lines_sorted[i]
            (x1_curr, y1_curr), (x2_curr, y2_curr) = current_line['points']
            
            # 确定当前线段的左右端点
            if x1_curr > x2_curr:
                x1_curr, x2_curr = x2_curr, x1_curr
                y1_curr, y2_curr = y2_curr, y1_curr
            
            current_angle = current_line['angle']
            
            # 计算当前线段的直线方程 (y = kx + b)
            # 对于水平线，k接近0，y接近常数
            if x2_curr != x1_curr:
                k_curr = (y2_curr - y1_curr) / (x2_curr - x1_curr)
                b_curr = y1_curr - k_curr * x1_curr
                is_vertical = False
            else:
                # 垂直线（应该很少，但处理一下）
                is_vertical = True
                x_const = x1_curr
            
            merged_group = [current_line]
            used_indices.add(i)
            
            # 寻找可以与当前线段合并的其他线段
            for j in range(i+1, len(lines_sorted)):
                if j in used_indices:
                    continue
                    
                other_line = lines_sorted[j]
                (x1_other, y1_other), (x2_other, y2_other) = other_line['points']
                
                # 确定其他线段的左右端点
                if x1_other > x2_other:
                    x1_other, x2_other = x2_other, x1_other
                    y1_other, y2_other = y2_other, y1_other
                
                other_angle = other_line['angle']
                
                # 1. 检查角度是否相近
                angle_diff = abs(current_angle - other_angle)
                if angle_diff > angle_threshold:
                    continue
                 
                # 检查端点距离
                # 计算当前线段与其他线段四个端点的最小距离
                distances = [
                    np.sqrt((x1_curr - x1_other)**2 + (y1_curr - y1_other)**2),
                    np.sqrt((x1_curr - x2_other)**2 + (y1_curr - y2_other)**2),
                    np.sqrt((x2_curr - x1_other)**2 + (y2_curr - y1_other)**2),
                    np.sqrt((x2_curr - x2_other)**2 + (y2_curr - y2_other)**2)
                ]
                end_distance = min(distances)
                if end_distance > end_threshold:
                    continue

                # 2. 检查Y坐标是否相近（对于接近水平的线）
                #y_diff = abs(current_line['y_avg'] - other_line['y_avg'])
                distances = [
                    abs(y1_curr - y1_other),
                    abs(y1_curr - y2_other),
                    abs(y2_curr - y1_other),
                    abs(y2_curr - y2_other)
                ]
                y_diff = min(distances)
                if y_diff > y_threshold:
                    continue
                
                # 3. 检查延长线之间的最近距离
                # line_distance = self._calculate_line_distance(current_line, other_line)
                # if line_distance <= max_extension_gap:
                #     merged_group.append(other_line)
                
                if not is_vertical:
                    # 计算其他线段的直线方程
                    if x2_other != x1_other:
                        k_other = (y2_other - y1_other) / (x2_other - x1_other)
                        b_other = y1_other - k_other * x1_other
                        
                        # 计算两条直线之间的平均距离
                        # 在两条线段的X重叠区域采样点计算距离
                        overlap_x_start = max(x1_curr, x1_other)
                        overlap_x_end = min(x2_curr, x2_other)
                        
                        if overlap_x_end > overlap_x_start:
                            # 在重叠区域采样
                            sample_points = np.linspace(overlap_x_start, overlap_x_end, num=10)
                            distances = []
                            for x in sample_points:
                                y_curr = k_curr * x + b_curr
                                y_other = k_other * x + b_other
                                distances.append(abs(y_curr - y_other))
                            
                            avg_distance = np.mean(distances)
                            
                            if avg_distance <= max_extension_gap:
                                merged_group.append(other_line)
                                used_indices.add(j)
                        else:
                            # 没有X重叠，计算延长线距离
                            # 计算从其他线段中点到当前直线的距离
                            mid_x_other = (x1_other + x2_other) / 2
                            mid_y_other = (y1_other + y2_other) / 2
                            
                            # 点到直线的距离公式
                            distance = abs(k_curr * mid_x_other - mid_y_other + b_curr) / np.sqrt(k_curr**2 + 1)
                            
                            if distance <= max_extension_gap:
                                merged_group.append(other_line)
                                used_indices.add(j)
                
                else:
                    # 垂直线的情况（较少见）
                    if abs(x_const - (x1_other + x2_other) / 2) <= max_extension_gap:
                        merged_group.append(other_line)
                        used_indices.add(j)
            
            # 合并组内的所有线段
            if merged_group:
                merged_line = self._merge_line_group_extended(merged_group)
                merged_lines.append(merged_line)
        
        return merged_lines

    def _merge_line_group_extended(self, line_group):
        """
        合并一组共线或近似共线的线段
        
        参数:
            line_group: 线段组列表
            
        返回:
            合并后的线段字典
        """
        if not line_group:
            return None
        
        # 收集所有端点和计算统计信息
        all_points = []
        all_angles = []
        all_lengths = []
        
        for line in line_group:
            all_points.extend(line['points'])
            all_angles.append(line['angle'])
            all_lengths.append(line['length'])
        
        # 计算加权平均角度（长线段权重更大）
        total_length = sum(all_lengths)
        if total_length > 0:
            weighted_angles = [angle * length for angle, length in zip(all_angles, all_lengths)]
            avg_angle = sum(weighted_angles) / total_length
        else:
            avg_angle = np.mean(all_angles)
        
        # 提取所有点的X和Y坐标
        all_x = [p[0] for p in all_points]
        all_y = [p[1] for p in all_points]
        
        # 对于接近水平的线（角度小于15度），使用线性回归拟合最佳水平线
        if avg_angle < 15:
            # 线性回归拟合所有点，得到最佳拟合直线
            A = np.vstack([all_x, np.ones(len(all_x))]).T
            k, b = np.linalg.lstsq(A, all_y, rcond=None)[0]
            
            # 使用拟合直线上的点作为合并后的线段端点
            min_x, max_x = min(all_x), max(all_x)
            start_y = k * min_x + b
            end_y = k * max_x + b
            
            start_point = (int(min_x), int(start_y))
            end_point = (int(max_x), int(end_y))
            
            new_length = np.sqrt((max_x - min_x)**2 + (end_y - start_y)**2)
            new_y_avg = (start_y + end_y) / 2
        else:
            # 对于非水平线，使用最小二乘法拟合直线
            A = np.vstack([all_x, np.ones(len(all_x))]).T
            k, b = np.linalg.lstsq(A, all_y, rcond=None)[0]
            
            # 找到所有点在拟合直线上的投影
            projected_points = []
            for x, y in zip(all_x, all_y):
                # 计算点到直线的投影
                proj_x = x
                proj_y = k * x + b
                projected_points.append((proj_x, proj_y))
            
            # 找到投影点的边界
            proj_x = [p[0] for p in projected_points]
            proj_y = [p[1] for p in projected_points]
            
            # 找到主方向上的起点和终点
            # 计算主成分方向
            points_array = np.array(projected_points)
            if len(points_array) > 1:
                # 找到主方向上的两个极端点
                diff = points_array - points_array.mean(axis=0)
                cov = diff.T @ diff / len(points_array)
                eigvals, eigvecs = np.linalg.eig(cov)
                
                # 主方向
                main_dir = eigvecs[:, np.argmax(eigvals)]
                
                # 在所有点上的投影值
                projections = points_array @ main_dir
                
                # 找到最小和最大投影对应的点
                min_idx = np.argmin(projections)
                max_idx = np.argmax(projections)
                
                start_point = tuple(points_array[min_idx].astype(int))
                end_point = tuple(points_array[max_idx].astype(int))
            else:
                start_point = (int(min(all_x)), int((min(all_y) + max(all_y)) / 2))
                end_point = (int(max(all_x)), int((min(all_y) + max(all_y)) / 2))
            
            new_length = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            new_y_avg = (start_point[1] + end_point[1]) / 2
        
        return {
            'points': [start_point, end_point],
            'length': new_length,
            'angle': avg_angle,
            'y_avg': new_y_avg
        }

    def _sample_depth_near_line(self, depth_filtered, line_points, offset=10):
        """
        在直线两侧采样深度，取最小值（因为产品侧深度小，背景侧深度大）
        
        参数:
            depth_filtered: 滤波后的深度图
            line_points: 直线上的像素点列表
            offset: 采样偏移距离（像素）
            
        返回:
            产品侧的平均深度（最小值），如果采样失败返回None
        """
        h, w = depth_filtered.shape
        
        # 存储两侧的深度值
        side1_depths = []  # 一侧（可能是产品侧）
        side2_depths = []  # 另一侧（可能是背景侧）
        
        for (x, y) in line_points:
            x_int, y_int = int(x), int(y)
            
            # 根据排序规则确定采样方向
            if self.sort_rule == SortRule.SORT_BY_Y_DESC:  # 从下到上扫描
                # 采样直线上方（Y更小）和下方（Y更大）
                sample_y_up = y_int - offset
                sample_y_down = y_int + offset
                
                # 采样上方点
                if 0 <= sample_y_up < h and 0 <= x_int < w:
                    curr_depth = depth_filtered[sample_y_up, x_int]
                    if (curr_depth != self.depth_invalid and 
                        self.feed_depth_min <= curr_depth <= self.feed_depth_max):
                        side1_depths.append(curr_depth)
                
                # 采样下方点
                if 0 <= sample_y_down < h and 0 <= x_int < w:
                    curr_depth = depth_filtered[sample_y_down, x_int]
                    if (curr_depth != self.depth_invalid and 
                        self.feed_depth_min <= curr_depth <= self.feed_depth_max):
                        side2_depths.append(curr_depth)
                        
            else:  # 从上到下扫描（SORT_BY_Y_ASC）
                # 采样直线下方（Y更大）和上方（Y更小）
                sample_y_down = y_int + offset
                sample_y_up = y_int - offset
                
                # 采样下方点
                if 0 <= sample_y_down < h and 0 <= x_int < w:
                    curr_depth = depth_filtered[sample_y_down, x_int]
                    if (curr_depth != self.depth_invalid and 
                        self.feed_depth_min <= curr_depth <= self.feed_depth_max):
                        side1_depths.append(curr_depth)
                
                # 采样上方点
                if 0 <= sample_y_up < h and 0 <= x_int < w:
                    curr_depth = depth_filtered[sample_y_up, x_int]
                    if (curr_depth != self.depth_invalid and 
                        self.feed_depth_min <= curr_depth <= self.feed_depth_max):
                        side2_depths.append(curr_depth)
        
        # 计算两侧的平均深度
        # avg_side1 = np.mean(side1_depths) if side1_depths else float('inf')
        # avg_side2 = np.mean(side2_depths) if side2_depths else float('inf')

        # print(f"  侧1平均深度: {avg_side1 if avg_side1 != float('inf') else 'None'}")
        # print(f"  侧2平均深度: {avg_side2 if avg_side2 != float('inf') else 'None'}")

        # 中值
        # sorted_side1_depths = np.sort(side1_depths)
        # avg_side1 = sorted_side1_depths[len(sorted_side1_depths)//2]
        
        # sorted_side2_depths = np.sort(side2_depths)
        # avg_side2 = sorted_side2_depths[len(sorted_side2_depths)//2]
        
        avg_side1 = np.median(side1_depths) if side1_depths else float('inf')
        avg_side2 = np.median(side2_depths) if side2_depths else float('inf')
        # print(f"  侧1中值深度: {avg_side1 if avg_side1 != float('inf') else 'None'}")
        # print(f"  侧2中值深度: {avg_side2 if avg_side2 != float('inf') else 'None'}")

        '''
        # 过滤：在均值上下偏差内重新筛选
        side1_filtered = []
        side2_filtered = []
        
        dev_threshold = avg_side1 / 20
        if side1_depths and avg_side1 != float('inf'):
            for d in side1_depths:
                if abs(d - avg_side1) <= dev_threshold:
                    side1_filtered.append(d)
                    
        dev_threshold = avg_side2 / 20
        if side2_depths and avg_side2 != float('inf'):
            for d in side2_depths:
                if abs(d - avg_side2) <= dev_threshold:
                    side2_filtered.append(d)
        
        # 再次计算过滤后的平均深度
        avg_side1 = np.mean(side1_filtered) if side1_filtered else float('inf')
        avg_side2 = np.mean(side2_filtered) if side2_filtered else float('inf')

        print(f"  侧1平均深度2: {avg_side1 if avg_side1 != float('inf') else 'None'}")
        print(f"  侧2平均深度2: {avg_side2 if avg_side2 != float('inf') else 'None'}")
        '''
        
        # 取最小值（产品侧的深度）
        if avg_side1 != float('inf') and avg_side2 != float('inf'):
            # 两侧都有有效数据，取较小值
            min_depth = min(avg_side1, avg_side2)
            # print(f"  两侧都有数据，取最小值: {min_depth:.1f}mm")
            return int(min_depth)
        elif avg_side1 != float('inf'):
            # print(f"  只有侧1有数据: {avg_side1:.1f}mm")
            return int(avg_side1)
        elif avg_side2 != float('inf'):
            # print(f"  只有侧2有数据: {avg_side2:.1f}mm")
            return int(avg_side2)
        else:
            # print("  两侧均无有效深度数据")
            return None

    '''  改用模块化版本: _detect_long_edge_horizontal_line
    # ==============================================================
    # ✅ ✅ ✅ 【直线检测算法完整版】支持多层产品分拣
    # ==============================================================
    def _depth_segment_find_horizontal_line(self, depth_img, rgb_img):
        regions = []
        h, w = depth_img.shape
        
        # 深度图预处理
        depth_filtered = cv2.medianBlur(depth_img, self.median_blur_kernel)
        depth_filtered = cv2.GaussianBlur(depth_filtered, (3,3), self.gaussian_sigma)
        
        # 限定ROI区域
        roi_x_start = max(self.feed_roi_x, 0)
        roi_x_end = min(self.feed_roi_x + self.feed_roi_w, w)
        roi_y_start = max(self.feed_roi_y, 0)
        roi_y_end = min(self.feed_roi_y + self.feed_roi_h, h)
        min_length = self.feed_min_length
        
        # 提取ROI区域
        roi_depth = depth_filtered[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_h, roi_w = roi_depth.shape
        
        # ===================== 多层产品识别 =====================
        product_height = self.product_height # 产品高度
        product_width = self.product_width # 产品宽度
        interval_height = self.interval_height # 间隔高度
        # feed_depth_start = self.feed_depth_start # 顶层产品初始高度
        tray_depth = self.tray_start_height # 托盘深度

        # 计算ROI内的有效深度
        roi_valid_depths = roi_depth[(roi_depth != self.depth_invalid) & 
                                     (roi_depth >= self.feed_depth_min) & 
                                     (roi_depth >= self.feed_depth_start) & 
                                     (roi_depth <= self.feed_depth_max)]
        
        current_layer = 0
        current_depth_min = self.feed_depth_min
        current_depth_max = self.feed_depth_max
        
        if len(roi_valid_depths) > 0:
            # 获取最浅深度（最上层产品）
            roi_min_depth = np.min(roi_valid_depths)
            print(f"ROI最浅深度: {roi_min_depth}mm")
            print(f"最上层产品深度: {feed_depth_start}mm")

            # 计算当前层
            # distance_from_tray = tray_depth - roi_min_depth
            # layer_thickness = product_height + interval_height

            # ===== 按深度值排序，取最浅的部分 =====
            # 深度值越小表示离相机越近（上层产品）
            
            # 按深度值从小到大排序（浅到深）
            sorted_depth_indices = np.argsort(roi_valid_depths)
            sorted_depths = roi_valid_depths[sorted_depth_indices]
            
            print(f"深度统计:")
            print(f"  总有效像素: {len(sorted_depths)}")
            print(f"  最浅深度: {sorted_depths[0]:.1f}mm")
            print(f"  最深深度: {sorted_depths[-1]:.1f}mm")
            
            # 取最上层的深度范围（根据产品数量比例）
            # 每层产品数的倒数的一半作为采样比例
            # product_count_per_layer = max(1, self.product_count_per_layer)
            # depth_ratio = 1.0 / (product_count_per_layer * 2)  # 每层产品数的倒数的一半
            # depth_ratio = min(depth_ratio, 0.2)  # 限制最大20%
            # depth_ratio = max(depth_ratio, 0.05)  # 限制最小5%
  
            # top_depth_count = int(len(sorted_depths) * depth_ratio)
            # top_depth_count = max(top_depth_count, 10)  # 至少取10个点
            top_depth_count = min(len(sorted_depths), 100) #固定100个点

            # 取深度最小的部分（最上层产品）
            top_depth_values = sorted_depths[:top_depth_count]
            
            # 计算上层区域的统计信息
            target_depth = np.mean(top_depth_values)
            target_depth_std = np.std(top_depth_values)
            
            # print(f"上层区域统计:")
            # print(f"  采样比例: {depth_ratio:.3f}")
            # print(f"  采样像素: {top_depth_count}")
            # print(f"  平均深度: {target_depth:.1f}mm")
            # print(f"  深度标准差: {target_depth_std:.1f}mm")
            
            # ===== 计算当前层：从最上层开始，自上而下计算 =====
            # 层厚 = 产品高度 + 间隔高度
            layer_thickness = product_height + interval_height
            
            # 计算当前深度相对于最上层的偏移
            depth_offset = target_depth - feed_depth_start
            
            # 如果偏移量为负（实际深度小于最上层深度），则视为第0层
            if depth_offset < 0:
                current_layer = 0
                print(f"警告: 检测深度({target_depth:.1f}mm)小于最上层深度({feed_depth_start:.1f}mm)，使用第0层")
            else:
                # 计算当前层号（0-based）
                current_layer = int(depth_offset // layer_thickness)
                
                # 确保层号不超过最大可能层数
                max_layer = 10  # 最大层数限制
                current_layer = min(current_layer, max_layer)
            
            print(f"深度偏移: {depth_offset:.1f}mm")
            print(f"层厚度: {layer_thickness:.1f}mm")
            print(f"当前层: 第{current_layer + 1}层 (层号: {current_layer})")

            # 计算当前层的深度范围（自上而下）
            # 第0层（最上层）：feed_depth_start ~ feed_depth_start + layer_thickness
            # 第1层：feed_depth_start + layer_thickness ~ feed_depth_start + 2*layer_thickness
            # 以此类推
            layer_top_depth = feed_depth_start + current_layer * layer_thickness      # 当前层下边界（深）
            layer_bottom_depth = feed_depth_start + (current_layer + 1) * product_height # layer_thickness   # 当前层上边界（浅）
            
            # 增加容差
            depth_tolerance = 0
            current_depth_min = layer_top_depth - depth_tolerance
            current_depth_max = layer_bottom_depth + depth_tolerance
  
            print(f"当前层: 第{current_layer + 1}层")
            print(f"层深度范围: {layer_top_depth:.1f} - {layer_bottom_depth:.1f}mm")
            print(f"检测深度范围: {current_depth_min:.1f} - {current_depth_max:.1f}mm")

            # # 计算当前层（基于目标深度）
            # distance_from_tray = tray_depth - target_depth
            # layer_thickness = product_height + interval_height

            # if distance_from_tray >= 0 and layer_thickness > 0:
            #     current_layer = int(distance_from_tray // layer_thickness)
                
            #     # 计算当前层的深度范围
            #     # layer_top_depth = tray_depth - (current_layer + 1) * layer_thickness     # 上边界（浅）
            #     # layer_bottom_depth = tray_depth - (current_layer + 0) * layer_thickness  # 下边界（深）
            #     layer_top_depth = target_depth - layer_thickness/2
            #     layer_bottom_depth = target_depth + layer_thickness/2

            #     # 增加容差
            #     depth_tolerance = 0
            #     current_depth_min = layer_top_depth - depth_tolerance
            #     current_depth_max = layer_bottom_depth + depth_tolerance
                
            #     print(f"当前层: 第{current_layer + 1}层")
            #     print(f"层深度范围: {layer_top_depth:.1f} - {layer_bottom_depth:.1f}mm")
            #     print(f"检测深度范围: {current_depth_min:.1f} - {current_depth_max:.1f}mm")

        debug = 1

        # 步骤1：创建深度边缘图
        max_depth = 1500
        roi_depth_clipped = roi_depth.copy()
        roi_depth_clipped[roi_depth_clipped > current_depth_max] = 0
        depth_normalized = roi_depth_clipped / 20
        depth_normalized = depth_normalized.astype(np.uint8)
        if debug == 1:
            cv2.imshow("depth_normalized", depth_normalized) 

        # 方法1：深度图边缘检测
        # Canny边缘检测
        thresh1 = self.feed_depth_thresh # 10
        thresh2 = thresh1 * 3 # 30
        edges_depth = cv2.Canny(depth_normalized, thresh1, thresh2)
        
        kernel = np.ones((5, 1), np.uint8)
        edges_depth = cv2.dilate(edges_depth, kernel, iterations=1)

        # 方法2：RGB图灰度边缘检测
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        gray_img_roi = gray_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end].copy()

        # RGB图的Canny边缘
        thresh1_rgb = 50
        thresh2_rgb = 30
        thresh3_rgb = 50 #100
        edges_rgb = cv2.Canny(gray_img_roi, thresh1_rgb, thresh2_rgb)
        edges_rgb[roi_depth_clipped > current_depth_max] = 0
        # edges_rgb[gray_img_roi < thresh3_rgb] = 0

        # 方法3：Sobel水平边缘检测（只保留水平方向的边缘）
        # 注意：Sobel算子计算的是梯度，需要进一步处理
        sobely = cv2.Sobel(gray_img_roi, cv2.CV_64F, 0, 1, ksize=3)
        sobely_abs = cv2.convertScaleAbs(sobely)
        
        # 对Sobel结果进行阈值化，只保留强边缘
        _, sobely_thresh = cv2.threshold(sobely_abs, 30, 255, cv2.THRESH_BINARY)
        # 可选：形态学操作，连接断开的边缘
        kernel_horizontal = np.ones((1, 5), np.uint8)  # 水平方向的核，用于连接水平边缘
        sobely_thresh = cv2.morphologyEx(sobely_thresh, cv2.MORPH_CLOSE, kernel_horizontal)

        if debug == 1:
            cv2.imshow("edges_depth", edges_depth)
            cv2.imshow("gray_img_roi", gray_img_roi)
            cv2.imshow("edges_rgb", edges_rgb)
            cv2.imshow("sobely_thresh", sobely_thresh)

        # ===================== 叠加多个边缘检测结果 =====================
        use_rgb = 1
        edges_combined = edges_depth
        if use_rgb == 1: # 方法1：取最大值（OR操作）
            edges_combined = cv2.bitwise_or(edges_depth, edges_rgb)
            # edges_combined = cv2.bitwise_or(edges_combined, sobely_thresh)

        elif use_rgb == 2: # 方法2：加权平均（可以根据需要调整权重）
            edges_combined = cv2.addWeighted(edges_depth, 0.5, edges_rgb, 0.5, 0)
            edges_combined = cv2.addWeighted(edges_combined, 0.7, sobely_thresh, 0.3, 0)

        edges_combined[gray_img_roi < thresh3_rgb] = 0

        # 直线最小长度阈值
        minLineLength = roi_w * 0.1

        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges_combined, 
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=10, 
                            minLineLength=minLineLength,
                            maxLineGap=10)
        
        # 筛选水平直线
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                if x2 != x1:
                    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                    
                    if angle < 15:  # 水平阈值
                        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        if length > minLineLength:
                            horizontal_lines.append({
                                'points': [(x1 + roi_x_start, y1 + roi_y_start), 
                                            (x2 + roi_x_start, y2 + roi_y_start)],
                                'length': length,
                                'angle': angle,
                                'y_avg': (y1 + y2) / 2 + roi_y_start,
                                'layer': current_layer
                            })
        
        # 显示原始直线
        if debug == 1:
            if len(depth_normalized.shape) == 2:
                depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

            for line in horizontal_lines:
                (x1, y1), (x2, y2) = line['points']
                cv2.line(depth_bgr, (int(x1-roi_x_start),int(y1-roi_y_start)), 
                        (int(x2-roi_x_start),int(y2-roi_y_start)), (0,255,0), 1)

            cv2.imshow("raw_line_filtered", depth_bgr)
            print(f"原始直线数量: {len(horizontal_lines)}")

        # 合并线段
        if horizontal_lines:
            horizontal_lines = self._merge_lines_by_extension(
                horizontal_lines, 
                end_threshold=50,
                max_extension_gap=20,
                angle_threshold=15,
                y_threshold=50
            )
            horizontal_lines = self._merge_lines_by_extension(
                horizontal_lines, 
                end_threshold=150,
                max_extension_gap=15,
                angle_threshold=10,
                y_threshold=50
            )
            horizontal_lines = self._merge_lines_by_extension(
                horizontal_lines, 
                end_threshold=250,
                max_extension_gap=15,
                angle_threshold=5,
                y_threshold=50
            )
            print(f"合并后线段数量: {len(horizontal_lines)}")

        # 构造region
        if debug == 1:
            if len(depth_normalized.shape) == 2:
                depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
            
        for target_line in horizontal_lines:
            (x1, y1), (x2, y2) = target_line['points']
            
            if debug == 1:
                color = (random.randint(40,255), random.randint(40,255), random.randint(40,255))
                cv2.line(depth_bgr, (int(x1-roi_x_start),int(y1-roi_y_start)), 
                        (int(x2-roi_x_start),int(y2-roi_y_start)), color, 2)

            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < min_length:
                print(f'skip line for distance = {distance}')
                continue

            # 计算几何中点
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            mid_x = roi_x_start + roi_w / 2 # x直接取图像中心
            edge_center_point = (mid_x, mid_y)
            
            # 提取直线上的像素点
            target_pixels = []
            target_depth_sum = 0.0
            valid_pixel_count = 0
            
            line_points = self._bresenham_line(x1, y1, x2, y2)
            
            depths = []
            for (x, y) in line_points:
                if (roi_x_start <= x < roi_x_end and 
                    roi_y_start <= y < roi_y_end):
                    
                    curr_depth = depth_filtered[int(y), int(x)]
                    if True:
                    # if (curr_depth != self.depth_invalid and 
                    #     self.feed_depth_min <= curr_depth <= self.feed_depth_max):
                        target_pixels.append((x, y))
                        depths.append(curr_depth)
                        target_depth_sum += curr_depth
                        valid_pixel_count += 1

            
            count_thresh = 10
            # if valid_pixel_count <= count_thresh:
            #     print(f'skip for valid_pixel_count = {valid_pixel_count}')

            if valid_pixel_count > count_thresh:
                print(f'valid_pixel_count = {valid_pixel_count}')
                
                # 计算像素中心
                cx = int(np.mean([p[0] for p in target_pixels]))
                cy = int(np.mean([p[1] for p in target_pixels]))
                pixel_center = (cx, cy)
                print(f'center = {cx}, {cy}')
                
                avg_depth = self._sample_depth_near_line(depth_filtered, target_pixels, offset=15)
                # if avg_depth < 400:
                #     avg_depth = self._sample_depth_near_line(depth_filtered, target_pixels, offset=30)

                if avg_depth is None:
                    print(f'skip line for avg_depth is None')
                    continue
                else:
                    print(f'avg depth = {avg_depth}')
                
                # 深度限定在当前层
                if avg_depth < current_depth_min or avg_depth > current_depth_max:
                    print(f'skip line for avg_depth = {avg_depth}, min = {current_depth_min}, max = {current_depth_max}')
                    continue

                area = valid_pixel_count
                
                # 旋转矩形
                pts = np.array(target_pixels, dtype=np.int32).reshape((-1,1,2))
                rotated_rect = cv2.minAreaRect(pts)
                rotate_angle = round(rotated_rect[2], 2)
                print(f'rotate_angle = {rotate_angle}')
                if rotate_angle > 45:
                    rotate_angle -= 90
                if rotate_angle > 45:
                    rotate_angle -= 90
                print(f'angle = {rotate_angle}')

                # 世界坐标
                world_xyz = self._pixel2world(edge_center_point, avg_depth)

                # 抓取位置的偏移坐标
                x, y, z = world_xyz
                x += self.feed_offset_x
                y += self.feed_offset_y
                z += self.feed_offset_z
                world_xyz = (x, y, z)

                # 构造region（新增layer字段）
                regions.append({
                    "region_id": len(regions) + 1,  # 改为动态ID
                    "pixel_center": pixel_center,
                    "edge_center_point": edge_center_point,
                    "world_xyz": world_xyz,
                    "rotate_angle": rotate_angle,
                    "rotated_rect": rotated_rect,
                    "area": area,
                    "avg_depth": avg_depth,
                    "layer": current_layer,  # 新增：记录所在层
                    "line_y_avg": target_line['y_avg'],  # 新增：直线Y坐标
                    "line_points": [(int(x1), int(y1)), (int(x2), int(y2))],  # 新增：直线端点
                })
        
        if debug == 1:
            cv2.imshow("line", depth_bgr)

        # 排序
        self._sort_regions(regions)
        
        # 打印排序结果
        for i, r in enumerate(regions):
            print(f'排序后 {i}: layer={r["layer"]}, y={r["pixel_center"][1]}, depth={r["avg_depth"]:.1f}')

        # 根据产品宽度过滤相近Y坐标的直线
        regions_filter = []
        if regions and len(regions) > 0:
            start_y = regions[0]["world_xyz"][1]
            for r in regions:
                if abs(r["world_xyz"][1] - start_y) < product_width:
                    regions_filter.append(r)
            # 确保至少保留一条直线
            if not regions_filter and regions:
                regions_filter = [regions[0]]
        else:
            regions_filter = regions

        self.detected_regions = regions_filter
        return regions_filter
    '''

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

    def _template_match_find_horizontal_line(self, depth_img, rgb_img):
        """
        基于模板匹配的产品边缘检测（支持深度图和RGB图）
        
        参数:
            depth_img: 深度图（用于深度验证）
            rgb_img: RGB图（用于RGB模板匹配）
            
        返回:
            regions: 检测到的区域列表，格式与直线检测一致
        """
        regions = []
        h, w = depth_img.shape
        
        # 深度图预处理
        depth_filtered = cv2.medianBlur(depth_img, self.median_blur_kernel)
        depth_filtered = cv2.GaussianBlur(depth_filtered, (3,3), self.gaussian_sigma)
        
        # 限定ROI区域
        roi_x_start = max(self.feed_roi_x, 0)
        roi_x_end = min(self.feed_roi_x + self.feed_roi_w, w)
        roi_y_start = max(self.feed_roi_y, 0)
        roi_y_end = min(self.feed_roi_y + self.feed_roi_h, h)
        
        # ===================== 多层产品识别 =====================
        product_height = self.product_height
        product_width = self.product_width
        interval_height = self.interval_height
        feed_depth_start = self.feed_depth_start
        
        # 提取ROI深度区域
        roi_depth = depth_filtered[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # 计算当前层深度范围
        roi_valid_depths = roi_depth[(roi_depth != self.depth_invalid) & 
                                     (roi_depth >= self.feed_depth_min) & 
                                     (roi_depth >= self.feed_depth_start) & 
                                     (roi_depth <= self.feed_depth_max)]
        
        current_layer = 0
        current_depth_min = self.feed_depth_min
        current_depth_max = self.feed_depth_max
        
        if len(roi_valid_depths) > 0:
            # 获取最浅深度
            roi_min_depth = np.min(roi_valid_depths)
            print(f"ROI最浅深度: {roi_min_depth}mm")
            
            # 计算当前层
            layer_thickness = product_height + interval_height
            depth_offset = roi_min_depth - feed_depth_start
            
            if depth_offset < 0:
                current_layer = 0
            else:
                current_layer = int(depth_offset // layer_thickness)
            
            # 计算当前层深度范围
            layer_top_depth = feed_depth_start + current_layer * layer_thickness
            layer_bottom_depth = feed_depth_start + (current_layer + 1) * product_height # layer_thickness
            
            current_depth_min = layer_top_depth
            current_depth_max = layer_bottom_depth
            
            print(f"当前层: {current_layer + 1}, 深度范围: {current_depth_min:.1f} - {current_depth_max:.1f}mm")
        
        # ===================== 加载模板 =====================
        template_depth = None
        template_rgb = None
        template_type = getattr(self, 'template_type', 'depth')  # 'depth' 或 'rgb'
        template_path = getattr(self, 'template_path', None)
        
        if template_path is None or not os.path.exists(template_path):
            print(f"模板文件不存在: {template_path}")
            return regions
        
        # 根据模板类型加载
        if template_type == 'depth':
            # 加载深度图模板（16位）
            template_depth = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
            if template_depth is None or template_depth.dtype != np.uint16:
                print(f"深度模板加载失败或格式不正确: {template_path}")
                return regions
            print(f"深度模板加载成功: {template_depth.shape}")
        else:
            # 加载RGB模板（8位）
            template_rgb = cv2.imread(template_path)
            if template_rgb is None:
                print(f"RGB模板加载失败: {template_path}")
                return regions
            print(f"RGB模板加载成功: {template_rgb.shape}")
        
        # ===================== 模板匹配 =====================
        # 提取ROI区域
        roi_depth_roi = depth_filtered[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        rgb_roi = rgb_img[roi_y_start:roi_y_end, roi_x_start:roi_x_start + roi_x_end]
        
        # 存储匹配结果
        matches = []
        
        if template_type == 'depth':
            # ========== 深度图模板匹配 ==========
            # 将16位深度图转换为8位用于匹配
            template_depth_8u = self._depth_to_8u(template_depth)
            roi_depth_8u = self._depth_to_8u(roi_depth_roi)
            
            # 方法1：直接模板匹配
            result = cv2.matchTemplate(roi_depth_8u, template_depth_8u, cv2.TM_CCOEFF_NORMED)
            
            # 方法2：边缘特征匹配（推荐）
            # thresh1 = 7
            # thresh2 = 30
            # edges_template = cv2.Canny(template_depth_8u, thresh1, thresh2)
            # edges_roi = cv2.Canny(roi_depth_8u, thresh1, thresh2)
            # result = cv2.matchTemplate(edges_roi, edges_template, cv2.TM_CCOEFF_NORMED)
            
            # 方法3：深度差匹配（如果模板有深度信息）
            # 可以计算深度差图进行匹配
            
        else:
            # ========== RGB图模板匹配 ==========
            gray_template = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2GRAY)
            
            # 方法1：原始灰度图匹配
            # result = cv2.matchTemplate(gray_roi, gray_template, cv2.TM_CCOEFF_NORMED)
            
            # 方法2：边缘特征匹配（推荐）
            edges_template = cv2.Canny(gray_template, 50, 150)
            edges_roi = cv2.Canny(gray_roi, 50, 150)
            result = cv2.matchTemplate(edges_roi, edges_template, cv2.TM_CCOEFF_NORMED)
            
            # 方法3：多尺度匹配（可选）
            # matches = self._multi_scale_template_match(gray_roi, gray_template)
        
        # 获取匹配结果
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        template_threshold = getattr(self, 'template_threshold', 0.6)
        
        print(f"模板匹配分数: {max_val:.3f}")
        # debug = 1
        # if debug == 1:
        #     cv2.imshow("edges_template", edges_template)
        #     cv2.imshow("edges_roi", edges_roi)

        if max_val < template_threshold:
            print(f"模板匹配失败，分数 {max_val:.3f} < {template_threshold}")
            return regions
        
        # 计算匹配位置
        template_h, template_w = template_depth.shape[:2] if template_type == 'depth' else template_rgb.shape[:2]
        match_x = max_loc[0] + roi_x_start
        match_y = max_loc[1] + roi_y_start
        
        # 获取模板边缘偏移（抓取位置相对于模板顶部/底部的偏移）
        template_edge_offset = getattr(self, 'template_edge_offset', template_h // 2)
        
        # 根据排序规则确定抓取边缘
        if self.sort_rule == SortRule.SORT_BY_Y_ASC:
            # 从上到下：抓取顶部边缘
            edge_y = match_y + template_edge_offset
        else:
            # 从下到上：抓取底部边缘
            edge_y = match_y + template_h - template_edge_offset
        
        # ===================== 深度验证 =====================
        # 提取匹配区域内的深度信息
        match_roi_depth = depth_filtered[match_y:match_y+template_h, match_x:match_x+template_w]
        valid_depths = match_roi_depth[(match_roi_depth != self.depth_invalid) &
                                        (match_roi_depth >= current_depth_min) &
                                        (match_roi_depth <= current_depth_max)]
        
        if len(valid_depths) == 0:
            print("匹配区域内无有效深度数据")
            return regions
        
        avg_depth = np.median(valid_depths)
        
        # ===================== 多目标匹配（可选） =====================
        # 如果需要匹配多个产品，可以使用非极大值抑制
        max_matches = getattr(self, 'max_template_matches', 1)
        if max_matches > 1:
            matches = self._multi_template_match(result, template_w, template_h, 
                                                  template_threshold, max_matches,
                                                  roi_x_start, roi_y_start)
        else:
            matches = [(match_x, match_y, max_val)]
        
        # ===================== 构造region =====================
        for i, (mx, my, score) in enumerate(matches[:max_matches]):
            # 计算抓取边缘
            if self.sort_rule == SortRule.SORT_BY_Y_ASC:
                edge_y = my + template_edge_offset
            else:
                edge_y = my + template_h - template_edge_offset
            
            edge_center_point = (mx + template_w // 2, edge_y)
            pixel_center = (mx + template_w // 2, my + template_h // 2)
            
            # 提取该匹配区域的深度
            match_roi = depth_filtered[my:my+template_h, mx:mx+template_w]
            valid_depths = match_roi[(match_roi != self.depth_invalid) &
                                      (match_roi >= current_depth_min) &
                                      (match_roi <= current_depth_max)]
            
            if len(valid_depths) == 0:
                continue
            
            avg_depth = np.median(valid_depths)
            
            # 计算世界坐标
            world_xyz = self._pixel2world(edge_center_point, avg_depth)
            
            # 添加抓取偏移
            x, y, z = world_xyz
            x += self.feed_offset_x
            y += self.feed_offset_y
            z += self.feed_offset_z
            
            region = {
                "region_id": i + 1,
                "pixel_center": pixel_center,
                "edge_center_point": edge_center_point,
                "world_xyz": (x, y, z),
                "rotate_angle": 0.0,
                "rotated_rect": None,
                "bounding_rect": (mx, my, template_w, template_h),
                "area": template_w * template_h,
                "avg_depth": avg_depth,
                "layer": current_layer,
                "line_y_avg": edge_y,
                "line_points": [(mx, edge_y), (mx + template_w, edge_y)],
                "match_score": score,
                "match_type": template_type
            }
            
            regions.append(region)
        
        # 排序
        self._sort_regions(regions, self.sort_rule)
        
        print(f"模板匹配成功，找到 {len(regions)} 个目标")
        
        # 保存用于显示结果
        self.detected_regions = regions
        
        return regions
    
    def _depth_to_8u(self, depth_16u, depth_min=None, depth_max=None):
        """将16位深度图转换为8位灰度图"""
        depth_16u = depth_16u.copy()
        
        # 过滤无效深度
        valid_depth = depth_16u[(depth_16u != 0) & (depth_16u < 65535)]
        
        if len(valid_depth) == 0:
            return np.zeros(depth_16u.shape[:2], dtype=np.uint8)
        
        if depth_min is None:
            depth_min = np.percentile(valid_depth, 2)
        if depth_max is None:
            depth_max = np.percentile(valid_depth, 98)
        
        if depth_max <= depth_min:
            depth_max = depth_min + 1
        
        depth_16u = np.clip(depth_16u, depth_min, depth_max)
        depth_8u = ((depth_16u - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        return depth_8u
    
    def _multi_template_match(self, result, template_w, template_h, threshold, max_matches, roi_x_start, roi_y_start):
        """多目标模板匹配（使用非极大值抑制）"""
        matches = []
        h, w = result.shape
        
        # 复制结果矩阵
        result_copy = result.copy()
        
        for _ in range(max_matches):
            # 找到最大值位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_copy)
            
            if max_val < threshold:
                break
            
            # 计算实际图像坐标
            match_x = max_loc[0] + roi_x_start
            match_y = max_loc[1] + roi_y_start
            
            matches.append((match_x, match_y, max_val))
            
            # 抑制该区域周围的匹配
            # 在结果矩阵中，将当前匹配区域周围的值设为最小值
            x_start = max(0, max_loc[0] - template_w // 2)
            x_end = min(w, max_loc[0] + template_w // 2)
            y_start = max(0, max_loc[1] - template_h // 2)
            y_end = min(h, max_loc[1] + template_h // 2)
            
            result_copy[y_start:y_end, x_start:x_end] = 0
        
        return matches
    
    def _multi_scale_template_match(self, gray_roi, gray_template):
        """多尺度模板匹配"""
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        best_match = None
        best_score = -1
        
        for scale in scales:
            # 缩放模板
            new_w = int(gray_template.shape[1] * scale)
            new_h = int(gray_template.shape[0] * scale)
            
            if new_w > gray_roi.shape[1] or new_h > gray_roi.shape[0]:
                continue
            
            scaled_template = cv2.resize(gray_template, (new_w, new_h))
            
            # 匹配
            result = cv2.matchTemplate(gray_roi, scaled_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = (max_loc, new_w, new_h, scale)
        
        return best_match, best_score

    def _get_product_template(self):
        """
        获取产品模板
        
        可以从配置文件加载，也可以从图像中学习
        """
        # 方法1：从文件加载
        if hasattr(self, 'template_path') and os.path.exists(self.template_path):
            template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                return template
        
        # 方法2：使用配置的模板尺寸创建默认模板
        template_w = getattr(self, 'template_width', 100)
        template_h = getattr(self, 'template_height', 50)
        
        # 创建一个矩形模板（边缘特征）
        template = np.zeros((template_h, template_w), dtype=np.uint8)
        # 绘制矩形边缘（用于边缘匹配）
        cv2.rectangle(template, (0, 0), (template_w-1, template_h-1), 255, 2)
        # 添加一些内部特征
        cv2.line(template, (0, template_h//2), (template_w-1, template_h//2), 255, 1)
        
        return template

    # ==============================================================
    # 模块1：深度图预处理（通用）
    # ==============================================================
    def _preprocess_depth(self, depth_img, median_kernel=None, gaussian_sigma=None):
        """深度图预处理：中值滤波 + 高斯滤波"""
        if median_kernel is None:
            median_kernel = self.median_blur_kernel
        if gaussian_sigma is None:
            gaussian_sigma = self.gaussian_sigma
        
        depth_filtered = cv2.medianBlur(depth_img, median_kernel)
        depth_filtered = cv2.GaussianBlur(depth_filtered, (3,3), gaussian_sigma)
        return depth_filtered

    # ==============================================================
    # 模块2：ROI提取（通用）
    # ==============================================================
    def _extract_roi(self, image, roi_x, roi_y, roi_w, roi_h):
        """提取ROI区域"""
        h, w = image.shape[:2]
        x_start = max(roi_x, 0)
        x_end = min(roi_x + roi_w, w)
        y_start = max(roi_y, 0)
        y_end = min(roi_y + roi_h, h)
        roi = image[y_start:y_end, x_start:x_end]
        return roi, x_start, x_end, y_start, y_end
    
    # ==============================================================
    # 模块3：层识别（通用，支持参数传入）
    # ==============================================================
    def _detect_current_layer_general(self, roi_depth, start_height, product_height, 
                                        interval_height, depth_min, depth_max):
        """
        检测当前产品所在层数（通用版）
        
        参数:
            roi_depth: ROI区域的深度图
            start_height: 顶层产品初始高度
            product_height: 产品高度
            interval_height: 间隔高度
            depth_min: 最小有效深度
            depth_max: 最大有效深度
            
        返回:
            current_layer: 当前层号（0-based）
            current_depth_min: 当前层最小深度
            current_depth_max: 当前层最大深度
            target_depth: 目标深度
        """
        # 计算ROI内的有效深度
        roi_valid_depths = roi_depth[(roi_depth != self.depth_invalid) & 
                                     (roi_depth >= depth_min) & 
                                     (roi_depth >= start_height) & 
                                     (roi_depth <= depth_max)]
        
        current_layer = 0
        current_depth_min = depth_min
        current_depth_max = depth_max
        target_depth = start_height
        
        if len(roi_valid_depths) > 0:
            # 获取最浅深度（最上层产品）
            roi_min_depth = np.min(roi_valid_depths)
            print(f"ROI最浅深度: {roi_min_depth}mm")
            print(f"最上层产品深度: {start_height}mm")
            
            # 按深度值从小到大排序
            sorted_depth_indices = np.argsort(roi_valid_depths)
            sorted_depths = roi_valid_depths[sorted_depth_indices]
            
            print(f"深度统计: 总有效像素: {len(sorted_depths)}, 最浅: {sorted_depths[0]:.1f}mm, 最深: {sorted_depths[-1]:.1f}mm")
            
            # 取最上层的深度范围（固定取前100个点）
            top_depth_count = min(len(sorted_depths), 100)
            top_depth_values = sorted_depths[:top_depth_count]
            target_depth = np.mean(top_depth_values)
            
            # 计算当前层
            layer_thickness = product_height + interval_height
            depth_offset = target_depth - start_height
            
            if depth_offset < 0:
                current_layer = 0
                print(f"警告: 检测深度({target_depth:.1f}mm)小于最上层深度({start_height:.1f}mm)，使用第0层")
            else:
                current_layer = int(depth_offset // layer_thickness)
                max_layer = 10
                current_layer = min(current_layer, max_layer)
            
            print(f"深度偏移: {depth_offset:.1f}mm, 层厚度: {layer_thickness:.1f}mm")
            print(f"当前层: 第{current_layer + 1}层 (层号: {current_layer})")
            
            # 计算当前层的深度范围
            layer_top_depth = start_height + current_layer * layer_thickness
            layer_bottom_depth = start_height + (current_layer + 1) * product_height
            
            current_depth_min = layer_top_depth
            current_depth_max = layer_bottom_depth
          
            print(f"当前层深度范围: {layer_top_depth:.1f} - {layer_bottom_depth:.1f}mm")
        
        return current_layer, current_depth_min, current_depth_max, target_depth
    
     # ==============================================================
    # 模块4：边缘检测（通用，支持参数传入）
    # ==============================================================
    def _detect_edges_general(self, roi_depth, rgb_roi, current_depth_min, current_depth_max, 
                               depth_thresh=None, use_rgb=True):
        """
        多源边缘检测（通用版）
        
        参数:
            roi_depth: ROI深度图
            rgb_roi: ROI RGB图
            current_depth_min: 当前层最小深度
            current_depth_max: 当前层最大深度
            depth_thresh: 深度图Canny阈值
            use_rgb: 是否使用RGB边缘
            
        返回:
            edges_combined: 合并后的边缘图
            depth_normalized: 归一化后的深度图
        """
        if depth_thresh is None:
            depth_thresh = self.feed_depth_thresh
        
        # 深度图边缘检测
        depth_normalized = roi_depth / 20
        depth_normalized = depth_normalized.astype(np.uint8)
        
        thresh1 = depth_thresh
        thresh2 = thresh1 * 3
        edges_depth = cv2.Canny(depth_normalized, thresh1, thresh2)

        kernel = np.ones((5, 1), np.uint8)
        edges_depth = cv2.dilate(edges_depth, kernel, iterations=1)

        if use_rgb and rgb_roi is not None:
            # RGB图边缘检测
            gray_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2GRAY)
            thresh1_rgb = 50
            thresh2_rgb = 30
            edges_rgb = cv2.Canny(gray_roi, thresh1_rgb, thresh2_rgb)
            
            # 合并边缘
            edges_combined = cv2.bitwise_or(edges_depth, edges_rgb)
        else:
            edges_combined = edges_depth

        return edges_combined, depth_normalized
    
    # ==============================================================
    # 模块5：按方向检测直线（通用）
    # ==============================================================
    def _detect_lines_by_orientation(self, edges, roi_w, roi_h, roi_x_start, roi_y_start,
                                      min_length_ratio=0.1, 
                                      is_horizontal=True, angle_threshold=15):
        """
        按方向检测直线
        
        参数:
            edges: 边缘图
            roi_w, roi_h: ROI尺寸
            roi_x_start, roi_y_start: ROI起始坐标
            min_length_ratio: 最小长度比例
            is_horizontal: True检测水平线，False检测垂直线
            angle_threshold: 角度阈值（度）
            
        返回:
            lines: 直线列表
        """
        if is_horizontal:
            min_line_length = roi_w * min_length_ratio
        else:
            min_line_length = roi_h * min_length_ratio
        
        lines_hough = cv2.HoughLinesP(edges, 
                                      rho=1, 
                                      theta=np.pi/180, 
                                      threshold=10, 
                                      minLineLength=min_line_length,
                                      maxLineGap=10)
        
        detected_lines = []
        if lines_hough is not None:
            for line in lines_hough:
                x1, y1, x2, y2 = line[0]
                
                if x2 != x1 or y2 != y1:
                    if is_horizontal:
                        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                    else:
                        angle = np.arctan2(abs(x2 - x1), abs(y2 - y1)) * 180 / np.pi
                    
                    if angle < angle_threshold:
                        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        if length > min_line_length:
                            line_info = {
                                'points': [(x1 + roi_x_start, y1 + roi_y_start), 
                                          (x2 + roi_x_start, y2 + roi_y_start)],
                                'length': length,
                                'angle': angle
                            }
                            if is_horizontal:
                                line_info['y_avg'] = (y1 + y2) / 2 + roi_y_start
                            else:
                                line_info['x_avg'] = (x1 + x2) / 2 + roi_x_start
                            
                            detected_lines.append(line_info)
        
        direction_name = "水平" if is_horizontal else "垂直"
        print(f"原始{direction_name}直线数量: {len(detected_lines)}")
        return detected_lines
    
    # ==============================================================
    # 模块6：线段合并（通用）
    # ==============================================================
    def _merge_lines_general(self, lines, is_horizontal=True):
        """合并共线的线段"""
        if not lines:
            return lines
        
        y_threshold_value = 50 if is_horizontal else 50
        
        # 多次合并
        merged = self._merge_lines_by_extension(
            lines, 
            end_threshold=50,
            max_extension_gap=20,
            angle_threshold=15,
            y_threshold=y_threshold_value
        )
        merged = self._merge_lines_by_extension(
            merged, 
            end_threshold=150,
            max_extension_gap=15,
            angle_threshold=10,
            y_threshold=y_threshold_value
        )
        merged = self._merge_lines_by_extension(
            merged, 
            end_threshold=250,
            max_extension_gap=15,
            angle_threshold=5,
            y_threshold=y_threshold_value
        )
        
        print(f"合并后线段数量: {len(merged)}")
        return merged
    
    # ==============================================================
    # 模块7：从直线构造Region（通用，支持偏移参数）
    # ==============================================================
    def _build_region_from_line_general(self, target_line, depth_filtered, 
                                         roi_x_start, roi_x_end, roi_y_start, roi_y_end,
                                         min_length, current_depth_min, current_depth_max,
                                         offset_x=0, offset_y=0, offset_z=0,
                                         is_horizontal=True):
        """
        从直线构造Region（通用版）
        
        参数:
            target_line: 直线信息
            depth_filtered: 滤波后的深度图
            roi_x_start, roi_x_end, roi_y_start, roi_y_end: ROI边界
            min_length: 最小长度阈值
            current_depth_min, current_depth_max: 当前层深度范围
            offset_x, offset_y, offset_z: 抓取偏移
            is_horizontal: 是否为水平线
            
        返回:
            region: 区域信息
        """
        (x1, y1), (x2, y2) = target_line['points']
        
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if distance < min_length:
            print(f'skip line for distance = {distance}')
            return None
        
        # 计算几何中点
        if is_horizontal:
            mid_x = roi_x_start + (roi_x_end - roi_x_start) // 2
            mid_y = (y1 + y2) // 2
        else:
            mid_x = (x1 + x2) // 2
            mid_y = roi_y_start + (roi_y_end - roi_y_start) // 2
        
        edge_center_point = (mid_x, mid_y)
        
        # 提取直线上的像素点
        target_pixels = []
        line_points = self._bresenham_line(x1, y1, x2, y2)
        
        for (x, y) in line_points:
            if (roi_x_start <= x < roi_x_end and 
                roi_y_start <= y < roi_y_end):
                target_pixels.append((x, y))
        
        count_thresh = 10
        if len(target_pixels) <= count_thresh:
            return None
        
        # 计算像素中心
        cx = int(np.mean([p[0] for p in target_pixels]))
        cy = int(np.mean([p[1] for p in target_pixels]))
        pixel_center = (cx, cy)
        
        # 采样深度
        avg_depth = self._sample_depth_near_line(depth_filtered, target_pixels, offset=15)
        
        if avg_depth is None:
            print(f'skip line for avg_depth is None')
            return None
        
        # 深度限定在当前层
        if avg_depth < current_depth_min or avg_depth > current_depth_max:
            print(f'skip line for avg_depth = {avg_depth}')
            return None
        
        area = len(target_pixels)
        
        # 旋转矩形
        pts = np.array(target_pixels, dtype=np.int32).reshape((-1,1,2))
        rotated_rect = cv2.minAreaRect(pts)
        rotate_angle = round(rotated_rect[2], 2)
        if rotate_angle > 45:
            rotate_angle -= 90
        if rotate_angle > 45:
            rotate_angle -= 90
        
        # 世界坐标
        world_xyz = self._pixel2world(edge_center_point, avg_depth)
        
        # 抓取位置的偏移坐标
        x, y, z = world_xyz
        x += offset_x
        y += offset_y
        z += offset_z
        
        region = {
            "region_id": 0,
            "pixel_center": pixel_center,
            "edge_center_point": edge_center_point,
            "world_xyz": (x, y, z),
            "rotate_angle": rotate_angle,
            "rotated_rect": rotated_rect,
            "area": area,
            "avg_depth": avg_depth,
            "line_points": [(int(x1), int(y1)), (int(x2), int(y2))]
        }
        
        if is_horizontal:
            region["line_y_avg"] = target_line.get('y_avg', (y1 + y2) // 2)
        else:
            region["line_x_avg"] = target_line.get('x_avg', (x1 + x2) // 2)
        
        return region
    
    # ==============================================================
    # 模块8：根据产品宽度过滤区域（通用）
    # ==============================================================
    def _filter_regions_by_width(self, regions, product_width):
        """根据产品宽度过滤相近Y坐标的直线"""
        regions_filter = []
        if regions and len(regions) > 0:
            start_y = regions[0]["world_xyz"][1]
            for r in regions:
                if abs(r["world_xyz"][1] - start_y) < product_width:
                    regions_filter.append(r)
            if not regions_filter and regions:
                regions_filter = [regions[0]]
        else:
            regions_filter = regions
        return regions_filter

    # ==============================================================
    # 长边水平线检测（检测产品长边方向的水平边缘）
    # ==============================================================
    def _detect_long_edge_horizontal_line(self, depth_img, rgb_img):
        """
        检测产品长边的水平边缘（默认模式）
        
        产品是长条铝型材，长边在深度图中呈现为较长的水平线。
        
        参数:
            depth_img: 深度图
            rgb_img: RGB图
            
        返回:
            regions: 检测到的长边边缘区域列表
        """
 
        # 深度图预处理
        depth_filtered = self._preprocess_depth(depth_img)
        
        # 提取ROI区域
        roi_depth, roi_x_start, roi_x_end, roi_y_start, roi_y_end = self._extract_roi(
            depth_filtered, self.feed_roi_x, self.feed_roi_y, 
            self.feed_roi_w, self.feed_roi_h
        )
        
        # 提取RGB ROI
        rgb_roi = rgb_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # 检测当前层及深度范围
        # current_layer, current_depth_min, current_depth_max, target_depth = self._detect_current_layer_general(
        #     roi_depth,
        #     self.product_start_height,
        #     self.product_height,
        #     self.interval_height,
        #     self.feed_depth_min,
        #     self.feed_depth_max
        # )
        current_depth_min = self.feed_depth_min
        current_depth_max = self.feed_depth_max

        # 边缘检测
        edges_combined, depth_normalized = self._detect_edges_general(
            roi_depth, rgb_roi, current_depth_min, current_depth_max, self.feed_depth_thresh, use_rgb=True
        )
        
        # 显示边缘图（调试用）
        debug = getattr(self, 'debug_mode', True)
        if debug:
            cv2.imshow("long_edge_edges", edges_combined)
        
        # 检测水平线
        roi_w = roi_x_end - roi_x_start
        roi_h = roi_y_end - roi_y_start
        
        horizontal_lines = self._detect_lines_by_orientation(
            edges_combined, roi_w, roi_h, roi_x_start, roi_y_start, 
            min_length_ratio=0.1,
            is_horizontal=True,
            angle_threshold=15
        )
        
        # 合并线段
        if debug == 1:
            line_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
            color = (random.randint(40,255), random.randint(40,255), random.randint(40,255))
            for target_line in horizontal_lines:
                (x1, y1), (x2, y2) = target_line['points']
                cv2.line(line_bgr, (int(x1-roi_x_start),int(y1-roi_y_start)), (int(x2-roi_x_start),int(y2-roi_y_start)), color, 2)
                cv2.imshow("merged_line", line_bgr)

        # 构造Region
        regions = []
        min_length = self.feed_min_length
        
        for target_line in horizontal_lines:
            region = self._build_region_from_line_general(
                target_line, depth_filtered, 
                roi_x_start, roi_x_end, roi_y_start, roi_y_end,
                min_length, current_depth_min, current_depth_max,
                offset_x=self.feed_offset_x,
                offset_y=self.feed_offset_y,
                offset_z=self.feed_offset_z,
                is_horizontal=True
            )
            if region:
                region["edge_type"] = "long_edge"
                regions.append(region)
        
        # 排序
        self._sort_regions(regions, self.sort_rule)
        
        # 打印排序结果
        for i, r in enumerate(regions):
            print(f'长边边缘排序后 {i}: y={r.get("line_y_avg", 0):.1f}, depth={r["avg_depth"]:.1f}')
        
        # 根据产品宽度过滤
        regions_filter = self._filter_regions_by_width(regions, self.product_width)
        
        self.detected_regions = regions_filter
        return regions_filter

    # ==============================================================
    # 模块：从边缘图提取水平线段（基于轮廓，适用于短边缘）
    # ==============================================================
    def _extract_horizontal_lines_from_contours(self, edges, roi_x_start, roi_y_start, 
                                                  min_width=10, min_height=10, 
                                                  max_height=500, min_aspect_ratio=3.0,
                                                  dilate_iterations=0):
        """
        从边缘图中通过轮廓检测提取水平线段（适用于短边缘）
        
        该方法通过轮廓检测代替霍夫直线检测，对短边缘更稳定。
        
        参数:
            edges: 边缘图（二值图像）
            roi_x_start, roi_y_start: ROI起始坐标（用于转换坐标）
            min_width: 最小轮廓宽度（像素）
            min_height: 最小轮廓高度（像素）
            max_height: 最大轮廓高度（像素）
            min_aspect_ratio: 最小宽高比（宽/高）
            dilate_iterations: 竖直方向膨胀次数（连接断裂边缘）
            
        返回:
            horizontal_lines: 线段列表，每个元素为字典，包含:
                - 'points': [(x1, y1), (x2, y2)] 线段端点（底边）
                - 'length': 线段长度
                - 'angle': 线段角度（0度）
                - 'y_avg': 线段平均Y坐标
                - 'contour_area': 轮廓面积
                - 'bounding_rect': 外接矩形 (x, y, w, h)
        """
        horizontal_lines = []
        
        if edges is None or edges.size == 0:
            return horizontal_lines
        
        # 步骤1：竖直方向膨胀，连接断裂的边缘
        if dilate_iterations > 0:
            kernel_vertical = np.ones((1, 11), np.uint8)
            edges_processed = cv2.dilate(edges, kernel_vertical, iterations=dilate_iterations)
        else:
            edges_processed = edges.copy()
        
        # 步骤2：查找轮廓
        contours, hierarchy = cv2.findContours(edges_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"找到 {len(contours)} 个轮廓")
        
        # 步骤3：筛选轮廓并提取底边
        for contour in contours:
            # 获取外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 筛选条件
            if w < min_width:
                continue
            if h < min_height: # or h > max_height:
                continue
            
            # 检查宽高比
            # aspect_ratio = w / h if h > 0 else 0
            # if aspect_ratio < min_aspect_ratio:
            #     continue
            
            # 获取轮廓的底边Y坐标（外接矩形底边）
            bottom_y = y + h
            
            # 可选：使用轮廓的最低点作为底边
            # 找到轮廓中Y坐标最大的点
            # bottom_y = max([p[0][1] for p in contour])
            
            # 线段端点（底边）
            x1 = x
            x2 = x + w
            y1 = bottom_y
            y2 = bottom_y
            
            # 转换到全局坐标
            x1_global = x1 + roi_x_start
            x2_global = x2 + roi_x_start
            y_global = y1 + roi_y_start
            
            # 计算线段长度
            length = x2_global - x1_global
            
            # 计算平均Y坐标
            x_avg = (x1_global + x2_global) / 2
            y_avg = y_global
            
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            line_info = {
                'points': [(x1_global, y_global), (x2_global, y_global)],
                'length': length,
                'angle': 0.0,  # 水平线
                'x_avg': x_avg,
                'y_avg': y_avg,
                'contour_area': area,
                'bounding_rect': (x1_global, y_global - h, w, h),
                'contour': contour  # 保存轮廓用于调试
            }
            
            horizontal_lines.append(line_info)
        
        # 按Y坐标排序
        horizontal_lines.sort(key=lambda l: l['y_avg'])
        
        print(f"轮廓筛选后得到 {len(horizontal_lines)} 条水平线段")
        
        return horizontal_lines
    
     # ==============================================================
    # 模块：合并X坐标范围重叠的水平线段（基于轮廓）
    # ==============================================================
    def _merge_overlapping_lines(self, lines, y_threshold=10):
        """
        合并X坐标范围有重叠或相近的水平线段
        
        参数:
            lines: 线段列表，每个元素包含 'points' 和 'y_avg'
            y_threshold: Y坐标差异阈值（像素），小于此值认为在同一水平线上
            
        返回:
            合并后的线段列表
        """
        if not lines:
            return lines
        
        # 按Y坐标排序
        lines_sorted = sorted(lines, key=lambda l: l['y_avg'])
        
        merged_lines = []
        used = [False] * len(lines_sorted)
        
        for i in range(len(lines_sorted)):
            if used[i]:
                continue
            
            current = lines_sorted[i]
            current_x = current['x_avg']
            current_y = current['y_avg']
            current_x1 = current['points'][0][0]
            current_x2 = current['points'][1][0]
            
            # 确保 x1 < x2
            if current_x1 > current_x2:
                current_x1, current_x2 = current_x2, current_x1
            
            # 寻找可以合并的线段（Y坐标相近）
            group = [current]
            used[i] = True
            
            for j in range(len(lines_sorted)):
                if used[j] or i == j:
                    continue
                
                other = lines_sorted[j]
                other_y = other['y_avg']
                other_x1 = other['points'][0][0]
                other_x2 = other['points'][1][0]
                
                # 确保 x1 < x2
                if other_x1 > other_x2:
                    other_x1, other_x2 = other_x2, other_x1
                
                # 检查Y坐标是否相近
                if y_threshold > 0 and abs(current_y - other_y) > y_threshold:
                    continue
                
                # 检查X坐标范围是否有重叠
                # 重叠条件：两个区间 [x1, x2] 和 [ox1, ox2] 有交集
                has_overlap = not (current_x2 < other_x1 or other_x2 < current_x1)
                
                if has_overlap:
                    group.append(other)
                    used[j] = True
            
            # 合并组内所有线段
            if len(group) == 1:
                merged_lines.append(current)
            else:
                # 合并：取所有线段的最小X和最大X
                all_x1 = []
                all_x2 = []
                all_y = []
                all_areas = []
                
                for line in group:
                    x1, y1 = line['points'][0]
                    x2, y2 = line['points'][1]
                    if x1 > x2:
                        x1, x2 = x2, x1
                    all_x1.append(x1)
                    all_x2.append(x2)
                    all_y.append(line['y_avg'])
                    all_areas.append(line.get('contour_area', line['length']))
                
                min_x = min(all_x1)
                max_x = max(all_x2)
                min_y = min(all_y)
                max_y = max(all_y)
                
                # 加权平均Y（按面积加权）
                # total_area = sum(all_areas)
                # if total_area > 0:
                #     weighted_y = sum(y * area for y, area in zip(all_y, all_areas)) / total_area
                # else:
                #     weighted_y = np.mean(all_y)
                weighted_y = max_y

                # 计算合并后的长度
                merged_length = max_x - min_x
                
                # 计算总轮廓面积
                total_contour_area = sum(all_areas)
                
                merged_line = {
                    'points': [(min_x, weighted_y), (max_x, weighted_y)],
                    'length': merged_length,
                    'angle': 0.0,
                    'x_avg': (x1 + x2) / 2,
                    'y_avg': weighted_y,
                    "bounding_rect": (min_x, min_y, max_x - min_x, max_y - min_y),
                    'merged_count': len(group),
                    'merged_areas': all_areas,
                    'contour_area': total_contour_area,
                    'original_x_ranges': list(zip(all_x1, all_x2))
                }
                merged_lines.append(merged_line)
        
        print(f"X重叠合并后剩余 {len(merged_lines)} 条线段")
        
        # 打印合并详情
        for i, line in enumerate(merged_lines):
            if line.get('merged_count', 1) > 1:
                print(f"  线段 {i+1}: 合并了 {line['merged_count']} 个轮廓, X范围 [{line['points'][0][0]}, {line['points'][1][0]}]")
        
        return merged_lines

    # ==============================================================
    # 端面水平线检测（基于轮廓，优化版）
    # ==============================================================
    def _detect_end_edge_horizontal_line(self, depth_img, rgb_img):
        """
        检测产品端面的水平边缘（基于轮廓方法）
        
        产品是长条铝型材，端面在深度图中呈现为较短的水平线。
        使用轮廓检测代替霍夫直线检测，对短边缘更稳定。
        
        参数:
            depth_img: 深度图
            rgb_img: RGB图
            
        返回:
            regions: 检测到的端面边缘区域列表
        """
        regions = []
        h, w = depth_img.shape
        
        # ===================== 使用独立的端面检测配置 =====================
        # 过滤无效深度值
        invalid_depth_mask = (depth_img == self.depth_invalid) | \
                           (depth_img < self.end_depth_min) | \
                           (depth_img > self.end_depth_max)
        depth_filtered = depth_img.copy()                   
        depth_filtered[invalid_depth_mask] = 0

        # 深度图预处理
        depth_filtered = self._preprocess_depth(depth_filtered)
        
        # 提取端面检测ROI区域
        roi_depth, roi_x_start, roi_x_end, roi_y_start, roi_y_end = self._extract_roi(
            depth_filtered, self.end_roi_x, self.end_roi_y, 
            self.end_roi_w, self.end_roi_h
        )
        
        # 提取RGB ROI
        if rgb_img is not None:
            rgb_roi = rgb_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        else:
            rgb_roi = None
        
        # 检测当前层及深度范围（使用端面检测的产品参数）
        # current_layer, current_depth_min, current_depth_max, target_depth = self._detect_current_layer_general(
        #     roi_depth,
        #     self.end_start_height,
        #     self.end_product_height,
        #     self.end_interval_height,
        #     self.end_depth_min,
        #     self.end_depth_max
        # )
   
        # 边缘检测（使用端面检测的深度阈值）
        edges_combined, depth_normalized = self._detect_edges_general(
            roi_depth, rgb_roi, self.end_depth_min, self.end_depth_max, 
            self.end_depth_thresh, use_rgb=(rgb_roi is not None)
        )
        
        # 显示边缘图（调试用）
        debug = getattr(self, 'debug_mode', True)
        if debug:
            cv2.imshow("depth_filtered", depth_filtered)
            cv2.imshow("end_edge_edges", edges_combined)
        
        # ===================== 从轮廓提取水平线段 =====================
        horizontal_lines = self._extract_horizontal_lines_from_contours(
            edges_combined, 
            roi_x_start, roi_y_start,
            min_width=self.end_min_length,
            min_height=20,                      # 最小高度3像素
            max_height=500                     # 最大高度500像素
        )
        
        # if debug:
        #     contour_display = rgb_img.copy()
        #     cv2.imshow("contour", contour_display)

        # 合并重叠的线段
        if horizontal_lines:
            horizontal_lines = self._merge_overlapping_lines(
                horizontal_lines, 
                y_threshold=0,                # Y坐标差异阈值
            )
        
        # ===================== 构造Region =====================
        for target_line in horizontal_lines:
            # 获取线段信息
            (x1, y1), (x2, y2) = target_line['points']
            line_x_avg = target_line['x_avg']
            line_y_avg = target_line['y_avg']
            
            # 计算几何中点
            mid_x = (x1 + x2) // 2
            mid_y = int(line_y_avg)
            edge_center_point = (mid_x, mid_y)
            pixel_center = (mid_x, mid_y - 10)  # 产品中心约在边缘上方
            avg_depth = self.end_depth_min
            
            # # 提取线段附近的像素点用于深度采样
            # line_points = self._bresenham_line(x1, y1, x2, y2)
            # target_pixels = []
            
            # for (x, y) in line_points:
            #     if (roi_x_start <= x < roi_x_end and 
            #         roi_y_start <= y < roi_y_end):
            #         target_pixels.append((x, y))
            
            # if len(target_pixels) < 5:
            #     continue
            
            # # 采样深度
            # avg_depth = self._sample_depth_near_line(depth_filtered, target_pixels, offset=35)
            
            # if avg_depth is None:
            #     print(f'skip line for avg_depth is None')
            #     continue
            
            # # 深度限定在当前层
            # if avg_depth < current_depth_min or avg_depth > current_depth_max:
            #     print(f'skip line for avg_depth = {avg_depth}')
            #     continue
            
            # 计算世界坐标
            world_xyz = self._pixel2world(edge_center_point, avg_depth)
            
            # 添加抓取偏移
            x_w, y_w, z_w = world_xyz
            x_w += self.end_offset_x
            y_w += self.end_offset_y
            z_w += self.end_offset_z
            
            current_layer = 0 #没用
            # 构造region
            region = {
                "region_id": 0,
                "pixel_center": pixel_center,
                "edge_center_point": edge_center_point,
                "world_xyz": (x_w, y_w, z_w),
                "rotate_angle": 0.0,
                "rotated_rect": None,
                "bounding_rect": target_line.get('bounding_rect', None),
                "area": target_line.get('contour_area', target_line['length']),
                "avg_depth": avg_depth,
                "layer": current_layer,
                "line_x_avg": line_x_avg,
                "line_y_avg": line_y_avg,
                "line_points": [(x1, y1), (x2, y2)],
                "color": self._get_color_by_layer(current_layer),
                "edge_type": "end_edge"
            }
            
            regions.append(region)
        
        # 排序
        self._sort_regions(regions, SortRule.SORT_BY_X_ASC)
        
        # 根据产品宽度过滤（去重）
        regions_filter = []
        if regions:
            used_x = []
            used_y = []
            for region in regions:
                is_duplicate = False
                for used in used_x:
                    if abs(region["line_x_avg"] - used) < self.end_product_width:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    regions_filter.append(region)
                    used_x.append(region["line_x_avg"])
                    # used_y.append(region["line_y_avg"])
        
        # 更新ID
        for i, region in enumerate(regions_filter):
            region["region_id"] = i + 1
        
        # 打印结果
        print(f"端面检测完成，找到 {len(regions_filter)} 个有效端面")
        for i, r in enumerate(regions_filter):
            print(f'  端面 {i+1}: y={r["line_y_avg"]:.1f}, depth={r["avg_depth"]:.1f}mm')
        
        # 显示调试图像
        if True:
            depth_display = self.depth_pseudo_color(roi_depth)
            for region in regions_filter:
                # 绘制线段
                (x1, y1), (x2, y2) = region["line_points"]
                cv2.line(depth_display, 
                        (x1 - roi_x_start, y1 - roi_y_start), 
                        (x2 - roi_x_start, y2 - roi_y_start), 
                        region["color"], 3)
                # 绘制中心点
                cx, cy = region["pixel_center"]
                cv2.circle(depth_display, 
                          (cx - roi_x_start, cy - roi_y_start), 
                          5, region["color"], -1)

            if getattr(self, "debug_mode", True):
                cv2.imshow("end_edge_detection", depth_display)
        
        self.detected_regions = regions_filter
        return regions_filter
    '''
    # ==============================================================
    # 端面水平线检测（相机翻转90度，检测产品端面边缘）
    # ==============================================================
    def _detect_end_edge_horizontal_line(self, depth_img, rgb_img):
        """
        检测产品端面的水平边缘（相机翻转90度拍摄）
        
        产品是长条铝型材，端面在深度图中呈现为较短的水平线。
        
        参数:
            depth_img: 深度图
            rgb_img: RGB图
            
        返回:
            regions: 检测到的端面边缘区域列表
        """
        regions = []
        h, w = depth_img.shape
        
        # ===================== 使用独立的端面检测配置 =====================         
        # 过滤无效深度值
        invalid_depth_mask = (depth_img == self.depth_invalid) | \
                           (depth_img < self.end_depth_min) | \
                           (depth_img > self.end_depth_max)
        depth_filtered = depth_img.copy()                   
        depth_filtered[invalid_depth_mask] = 0
        cv2.imshow("depth_filtered", depth_filtered)

        # 深度图预处理
        depth_filtered = self._preprocess_depth(depth_filtered)
        
        # 提取端面检测ROI区域
        roi_depth, roi_x_start, roi_x_end, roi_y_start, roi_y_end = self._extract_roi(
            depth_filtered, self.end_roi_x, self.end_roi_y, 
            self.end_roi_w, self.end_roi_h
        )
  
        
        # 提取RGB ROI
        if rgb_img is not None:
            rgb_roi = rgb_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        else:
            rgb_roi = None
        
        # 检测当前层及深度范围（使用端面检测的产品参数）
        # current_layer, current_depth_min, current_depth_max, target_depth = self._detect_current_layer_general(
        #     roi_depth,
        #     self.end_start_height,
        #     self.end_product_height,
        #     self.end_interval_height,
        #     self.end_depth_min,
        #     self.end_depth_max
        # )
        current_depth_min = self.end_depth_min
        current_depth_max = self.end_depth_max

        # 边缘检测（使用端面检测的深度阈值）
        edges_combined, depth_normalized = self._detect_edges_general(
            roi_depth, rgb_roi, current_depth_min, current_depth_max, 
            self.end_depth_thresh, use_rgb=(rgb_roi is not None)
        )
        
        # 显示边缘图（调试用）
        debug = getattr(self, 'debug_mode', True)
        if debug:
            cv2.imshow("end_edge_edges", edges_combined)
            cv2.imshow("end_edge_depth_norm", depth_normalized)
        
        # 检测水平线（端面边缘是水平方向）
        roi_w = roi_x_end - roi_x_start
        roi_h = roi_y_end - roi_y_start
        
        horizontal_lines = self._detect_lines_by_orientation(
            edges_combined, roi_w, roi_h, roi_x_start, roi_y_start, 
            min_length_ratio=0.08,      # 端面边缘较短，降低比例
            is_horizontal=True,
            angle_threshold=15
        )
        
        # 合并线段
        if horizontal_lines:
            horizontal_lines = self._merge_lines_general(horizontal_lines, is_horizontal=True)
        
        
        # 构造Region（使用端面检测的偏移参数）
        min_length = self.end_min_length
        
        for target_line in horizontal_lines:
            region = self._build_region_from_line_general(
                target_line, depth_filtered, 
                roi_x_start, roi_x_end, roi_y_start, roi_y_end,
                min_length, current_depth_min, current_depth_max,
                offset_x=self.end_offset_x,
                offset_y=self.end_offset_y,
                offset_z=self.end_offset_z,
                is_horizontal=True
            )
            if region:
                region["edge_type"] = "end_edge"
                regions.append(region)
        
        # 排序
        self._sort_regions(regions, self.sort_rule)
        
        # 打印排序结果
        for i, r in enumerate(regions):
            print(f'端面边缘排序后 {i}: layer={r["layer"]}, y={r.get("line_y_avg", 0):.1f}, depth={r["avg_depth"]:.1f}')
        
        # 根据产品端面宽度过滤
        regions_filter = self._filter_regions_by_width(regions, self.end_product_width)
        
        self.detected_regions = regions_filter
        return regions_filter
    '''
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

    # ===================== 【修改后✅核心】下料算法：每层X1个，沿Y竖直并排 =====================
    # ===================== 【简化优化✅核心】下料算法：单层深度比较判断 =====================
    def _unload_check(self, depth_img):
        """下料算法：基于单层深度比较判断产品放置状态"""
        exists_flag = DetectStatus.NOTHING
        coords = [0.0, 0.0, 0.0, 0.0]
        
        # 1. 预处理深度图
        depth_filtered = cv2.medianBlur(depth_img, self.median_blur_kernel)
        depth_filtered = cv2.GaussianBlur(depth_filtered, (3,3), self.gaussian_sigma)
        
        # 2. 限定下料ROI范围
        h, w = depth_img.shape
        roi_x_start = max(self.unload_roi_x, 0)
        roi_x_end = min(self.unload_roi_x + self.unload_roi_w, w)
        roi_y_start = max(self.unload_roi_y, 0)
        roi_y_end = min(self.unload_roi_y + self.unload_roi_h, h)
        
        # 3. 生成单层理论产品区域坐标（所有层坐标都一样）
        item_regions = []
        region_depths = []  # 存储每个区域的平均深度
        
        # 层号固定为0（只考虑一层）
        layer_idx = 0
        layer_depth = self.feed_depth_min  # 第一层基准深度
        
        # 生成该层所有产品的区域
        for item_idx in range(self.unload_item_count_per_layer):
            # 计算产品区域坐标
            left = roi_x_start
            top = roi_y_start + item_idx * (self.unload_item_height + self.unload_item_interval)
            right = left + self.unload_item_width
            bottom = top + self.unload_item_height
            
            # 确保在ROI范围内
            left = max(left, roi_x_start)
            top = max(top, roi_y_start)
            right = min(right, roi_x_end)
            bottom = min(bottom, roi_y_end)
            
            if right <= left or bottom <= top:
                continue  # 无效区域
                
            # 提取产品区域的深度值
            region_depth = depth_filtered[top:bottom, left:right]
            
            # 过滤无效深度值
            valid_mask = (region_depth != self.depth_invalid) & \
                        (region_depth >= self.feed_depth_min) & \
                        (region_depth <= self.feed_depth_max)
            valid_depths = region_depth[valid_mask]
            
            # 计算平均深度
            if len(valid_depths) > 0:
                mean_depth = np.mean(valid_depths)
                valid_ratio = len(valid_depths) / region_depth.size
            else:
                mean_depth = layer_depth  # 默认值
                valid_ratio = 0
            
            # 存储区域信息
            region_info = {
                "layer_idx": layer_idx,
                "item_idx": item_idx,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "center_x": (left + right) // 2,
                "center_y": (top + bottom) // 2,
                "target_depth": layer_depth,
                "mean_depth": mean_depth,
                "valid_ratio": valid_ratio,
                "is_valid": valid_ratio > 0.3  # 有效像素比例阈值
            }
            
            item_regions.append(region_info)
            if region_info["is_valid"]:
                region_depths.append(mean_depth)
        
        # 4. 判断放置状态
        empty_regions = []
        
        if len(region_depths) > 0:
            # 计算所有有效区域的平均深度和标准差
            all_mean_depth = np.mean(region_depths)
            all_std_depth = np.std(region_depths) if len(region_depths) > 1 else 0
            
            print(f"深度统计: 平均值={all_mean_depth:.1f}, 标准差={all_std_depth:.1f}")
            
            # 判断规则：
            # 1. 如果所有区域深度标准差很小（< 阈值），说明深度一致
            # 2. 如果某个区域深度明显大于平均深度，说明该区域为空
            
            depth_std_threshold = self.unload_depth_threshold # 15  # 深度标准差阈值
            
            if all_std_depth < depth_std_threshold:
                # 情况1: 所有区域深度差不多
                print("状态: 所有区域深度一致")
                
                # 判断是全部空还是全部有产品
                if True: #all_mean_depth > layer_depth + 30:  # 深度明显大于基准，说明全部空
                    print("结论: 全部区域为空")
                    # 所有区域都为空，根据排序规则选择第一个
                    empty_regions = [r for r in item_regions if r["is_valid"]]
                '''
                else:
                    print("结论: 全部区域已放置产品")
                    # 如果全部已放置，根据排序规则返回最上面或最下面的边缘
                    if self.sort_rule == SortRule.SORT_BY_Y_ASC:
                        # 从上到下：选择最上面的区域
                        item_regions.sort(key=lambda r: r["top"])
                        target_region = item_regions[0]
                        # 使用顶部边缘中心
                        edge_x = target_region["center_x"]
                        edge_y = target_region["top"]
                        edge_center = (edge_x, edge_y)
                    elif self.sort_rule == SortRule.SORT_BY_Y_DESC:
                        # 从下到上：选择最下面的区域
                        item_regions.sort(key=lambda r: r["bottom"], reverse=True)
                        target_region = item_regions[0]
                        # 使用底部边缘中心
                        edge_x = target_region["center_x"]
                        edge_y = target_region["bottom"]
                        edge_center = (edge_x, edge_y)
                    else:
                        # 默认使用第一个区域的中心
                        target_region = item_regions[0]
                        edge_center = (target_region["center_x"], target_region["center_y"])
                    
                    # 计算世界坐标
                    world_xyz = self._pixel2world(edge_center, target_region["mean_depth"])
                    x, y, z = world_xyz
                    r = 0.0
                    
                    # 叠加工具坐标偏移
                    x += self.tool_coord_x
                    y += self.tool_coord_y
                    z += self.tool_coord_z
                    r += self.tool_coord_r
                    
                    coords = [x, y, z, r]
                    exists_flag = DetectStatus.EXIST
                    
                    print(f"返回坐标: 区域{target_region['item_idx']}, 边缘({edge_x}, {edge_y})")
                    return exists_flag, coords
                '''
            else:
                # 情况2: 深度有差异，找出空区域
                print("状态: 区域深度有差异")
                
                # 空区域判断：深度比平均深度大一定阈值
                depth_diff_threshold = depth_std_threshold # 25  # 深度差异阈值
                
                for region in item_regions:
                    if not region["is_valid"]:
                        # 无效区域视为空
                        empty_regions.append(region)
                        continue
                        
                    depth_diff = region["mean_depth"] - all_mean_depth
                    if depth_diff > depth_diff_threshold:
                        # 深度明显大于平均，说明该区域为空
                        empty_regions.append(region)
                        print(f"  区域{region['item_idx']}: 深度={region['mean_depth']:.1f}, "
                            f"差异={depth_diff:.1f} > 阈值, 判断为空")
        
        else:
            # 没有有效深度数据，所有区域都视为空
            print("状态: 无有效深度数据，所有区域视为空")
            empty_regions = item_regions
        
        # 5. 如果有空区域，根据排序规则选择第一个空位置
        if empty_regions:
            exists_flag = DetectStatus.EXIST
            
            # 根据排序规则对空区域排序
            if self.sort_rule == SortRule.SORT_BY_Y_ASC:
                # 从上到下：选择Y坐标最小的空位置（最上面）
                empty_regions.sort(key=lambda r: r["top"])
                target_region = empty_regions[0]
                # 使用顶部边缘中心
                edge_x = target_region["center_x"]
                edge_y = target_region["top"]
                edge_center = (edge_x, edge_y)
            elif self.sort_rule == SortRule.SORT_BY_Y_DESC:
                # 从下到上：选择Y坐标最大的空位置（最下面）
                empty_regions.sort(key=lambda r: r["bottom"], reverse=True)
                target_region = empty_regions[0]
                # 使用底部边缘中心
                edge_x = target_region["center_x"]
                edge_y = target_region["bottom"]
                edge_center = (edge_x, edge_y)
            else:
                # 默认按物品索引排序
                empty_regions.sort(key=lambda r: r["item_idx"])
                target_region = empty_regions[0]
                edge_center = (target_region["center_x"], target_region["center_y"])
            
            print(f"选择空区域: 区域{target_region['item_idx']}, "
                f"深度={target_region['mean_depth']:.1f}, "
                f"边缘({edge_x}, {edge_y})")
            
            # 计算世界坐标
            # 空位置使用目标深度，而不是实际深度
            target_depth = target_region["target_depth"]
            world_xyz = self._pixel2world(edge_center, target_depth)
            x, y, z = world_xyz
            r = 0.0
            
            # 叠加工具坐标偏移
            x += self.tool_coord_x
            y += self.tool_coord_y
            z += self.tool_coord_z
            r += self.tool_coord_r
            
            coords = [x, y, z, r]
        
        else:
            exists_flag = DetectStatus.NOTHING
            print("状态: 未找到空区域")
        
        return exists_flag, coords

    # ===================== 【对外接口2 - 检测接口】核心 =====================
    def detect(self, ptype, rgb_img, depth_img):
        try:
            if not self.config_loaded:
                return {"code":-1, "err_msg":"请先调用初始化函数加载配置"}
            if depth_img is None or depth_img.dtype != np.uint16:
                return {"code":-2, "err_msg":"深度图格式错误，必须是CV_16UC1单通道格式"}
            if ptype < PType.MATERIAL_CHECK or ptype > PType.FEED_END_CHECK:
                return {"code":-3, "err_msg":"ptype类型错误，仅支持1/2/3/4/5"}
            
            coords = [0.0, 0.0, 0.0, 0.0]
            exists_flag = DetectStatus.UNKNOWN
            regions = []

            # 分支处理不同检测类型
            if ptype == PType.MATERIAL_CHECK: # 物料缓存台
                exists_flag, coords = self._material_check(depth_img)
            elif ptype == PType.IRON_CHIP_CHECK: # 铁屑
                exists_flag, coords = self._judge_detect_result(regions, ptype, rgb_img)
            elif ptype == PType.FEED_CHECK: # 上料
                # regions = self._depth_segment_find_horizontal_line(depth_img, rgb_img)
                # regions = self._template_match_find_horizontal_line(depth_img, rgb_img)
                regions =  self._detect_long_edge_horizontal_line(depth_img, rgb_img)
                if not regions:
                    exists_flag = DetectStatus.NOTHING
                elif  self.feed_edge_index >= len(regions):
                     # 索引超出范围，返回最后一个
                    print(f"警告: 指定的索引 {self.feed_edge_index} 超出范围 (0-{len(regions)-1})，使用最后一个")
                    main_region = regions[-1]
                    exists_flag = DetectStatus.EXIST
                    x, y, z = main_region["world_xyz"]
                    r = main_region["rotate_angle"]
                    x += self.tool_coord_x
                    y += self.tool_coord_y
                    z += self.tool_coord_z
                    r += self.tool_coord_r
                    coords = [x, y, z, r]
                else:
                    exists_flag = DetectStatus.EXIST
                    # 提取指定索引的区域坐标
                    main_region = regions[self.feed_edge_index]
                    x, y, z = main_region["world_xyz"]
                    r = main_region["rotate_angle"]
                    x += self.tool_coord_x
                    y += self.tool_coord_y
                    z += self.tool_coord_z
                    r += self.tool_coord_r
                    coords = [x, y, z, r]
                    
                print(f"返回第 {self.feed_edge_index} 个区域，共检测到 {len(regions)} 个区域")
            elif ptype == PType.FEED_END_CHECK: # 上料前检测端点
                regions =  self._detect_end_edge_horizontal_line(depth_img, None)
                if not regions:
                    exists_flag = DetectStatus.NOTHING
                else:
                    exists_flag = DetectStatus.EXIST
                    # 提取指定索引的区域坐标
                    main_region = regions[self.feed_edge_index]
                    x, y, z = main_region["world_xyz"]
                    r = main_region["rotate_angle"]
                    x += self.tool_coord_x
                    y += self.tool_coord_y
                    z += self.tool_coord_z
                    r += self.tool_coord_r
                    coords = [x, y, z, r]
                    
                print(f"返回 {len(regions)} 个区域")
            elif ptype == PType.UNLOAD_CHECK: # 下料
                exists_flag, coords = self._unload_check(depth_img)

            return {
                "code":0,
                "result":{
                    "ptype":ptype,
                    "coords":coords,
                    "exists": exists_flag
                },
                "err_msg":""
            }
        except Exception as e:
            logger.info(traceback.format_exc())
            print(traceback.format_exc())
            return {"code":-99, "err_msg":f"检测异常: {str(e)}"}

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

    def _get_color_by_layer(self, layer):
        """根据层数分配颜色"""
        colors = [
            (0, 0, 255),      # 红色
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 蓝色
            (0, 255, 255),    # 黄色
            (255, 0, 255),    # 紫色
            (255, 255, 0),    # 青色
        ]
        return colors[layer % len(colors)]
        
    # ===================== 绘制函数 - 可视化水平直线 + 原标注 =====================
    # 废弃版本
    def draw_result(self, rgb, detect_res):
        draw_img = rgb.copy()
        if detect_res["code"] != 0 or not detect_res["result"]:
            cv2.putText(draw_img, "DETECT ERR", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return draw_img
        
        res = detect_res["result"]
        ptype = res["ptype"]
        coords = res["coords"]
        exists_flag = res["exists"]
        x,y,z,r = coords

        status_text = "EXIST" if exists_flag == DetectStatus.EXIST else "NOTHING"
        status_color = (0,255,0) if exists_flag == DetectStatus.EXIST else (0,0,255)
        cv2.putText(draw_img, f"STATUS: {status_text}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # 检测类型
        type_dict = {1: "Material", 2: "Feed", 3: "Unload", 4: "Iron", 5: "End" }
        cv2.putText(draw_img, f"TYPE: {type_dict[ptype]}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # 绘制坐标信息
        coord_text = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f} R:{r:.1f}"
        cv2.putText(draw_img, coord_text, (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # 绘制对应ROI框
        if ptype == PType.MATERIAL_CHECK:
            # 物料缓存台ROI（绿色）
            cv2.rectangle(draw_img, (self.material_roi_x, self.material_roi_y), 
                         (self.material_roi_x+self.material_roi_w, self.material_roi_y+self.material_roi_h), 
                         (0,255,0), 2)
            cv2.putText(draw_img, "Material ROI", (self.material_roi_x+5, self.material_roi_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        elif ptype == PType.FEED_CHECK:
            # 上料ROI（蓝色）
            cv2.rectangle(draw_img, (self.feed_roi_x, self.feed_roi_y), 
                         (self.feed_roi_x+self.feed_roi_w, self.feed_roi_y+self.feed_roi_h), 
                         (255,0,0), 2)
            cv2.putText(draw_img, "Feed ROI", (self.feed_roi_x+5, self.feed_roi_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        elif ptype == PType.UNLOAD_CHECK:
            # 下料ROI（黄色）
            cv2.rectangle(draw_img, (self.unload_roi_x, self.unload_roi_y), 
                         (self.unload_roi_x+self.unload_roi_w, self.unload_roi_y+self.unload_roi_h), 
                         (0,255,255), 2)
            cv2.putText(draw_img, "Unload ROI", (self.unload_roi_x+5, self.unload_roi_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        elif ptype == PType.IRON_CHIP_CHECK:
            # YOLO检测ROI（紫色）
            cv2.rectangle(draw_img, (self.yolo_roi_x, self.yolo_roi_y), 
                         (self.yolo_roi_x+self.yolo_roi_w, self.yolo_roi_y+self.yolo_roi_h), 
                         (255,0,255), 2)
            cv2.putText(draw_img, "YOLO ROI", (self.yolo_roi_x+5, self.yolo_roi_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            # 绘制YOLO铁屑检测框
            for (x1, y1, w, h, conf) in self.detect_iron_chips:
                cv2.rectangle(draw_img, (x1, y1), (x1+w, y1+h), (0,0,255), 2)
                cv2.putText(draw_img, f"{conf}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        return draw_img

    # ===================== 【核心优化 ✅ 重点】叠加 YOLO铁屑框+置信度+旋转矩形框+中心点+ROI框+所有信息 =====================
    def draw_result_with_rotated_box(self, rgb, detect_res):
        draw_img = rgb.copy()
        # 检测异常的情况 加双重判断
        if detect_res["code"] != 0 or not detect_res["result"]:
            cv2.putText(draw_img, "DETECT ERR", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if detect_res["err_msg"]:
                cv2.putText(draw_img, detect_res["err_msg"][:15], (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            return draw_img
        
        res = detect_res["result"]
        ptype = res["ptype"]
        coords = res["coords"]
        exists_flag = res["exists"]
        x,y,z,r = coords

        # 1. 绘制原有所有信息：状态、类型、坐标
        status_text = "EXIST" if exists_flag == DetectStatus.EXIST else "NOTHING"
        status_color = (0,255,0) if exists_flag == DetectStatus.EXIST else (0,0,255)
        cv2.putText(draw_img, f"STATUS: {status_text}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # 检测类型
        type_dict = {1: "Material", 2: "Feed", 3: "Unload", 4: "Iron", 5: "End"}
        cv2.putText(draw_img, f"TYPE: {type_dict[ptype]}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # 绘制坐标信息
        coord_text = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f} R:{r:.1f}"
        cv2.putText(draw_img, coord_text, (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # 显示检测到的区域数量
        if len(self.detected_regions) > 0:
            cv2.putText(draw_img, f"Regions: {len(self.detected_regions)}", (20,150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # 2. 绘制对应ROI框
        roi_info = {
            PType.MATERIAL_CHECK: ("Material ROI", (self.material_roi_x, self.material_roi_y), 
                                  (self.material_roi_x+self.material_roi_w, self.material_roi_y+self.material_roi_h), (0,255,0)),
            PType.FEED_CHECK: ("Feed ROI", (self.feed_roi_x, self.feed_roi_y), 
                              (self.feed_roi_x+self.feed_roi_w, self.feed_roi_y+self.feed_roi_h), (255,0,0)),
            PType.UNLOAD_CHECK: ("Unload ROI", (self.unload_roi_x, self.unload_roi_y), 
                                (self.unload_roi_x+self.unload_roi_w, self.unload_roi_y+self.unload_roi_h), (0,255,255)),
            PType.IRON_CHIP_CHECK: ("YOLO ROI", (self.yolo_roi_x, self.yolo_roi_y), 
                                   (self.yolo_roi_x+self.yolo_roi_w, self.yolo_roi_y+self.yolo_roi_h), (255,0,255))
        }
        if ptype in roi_info:
            roi_name, roi_start, roi_end, roi_color = roi_info[ptype]
            # 绘制ROI框
            cv2.rectangle(draw_img, roi_start, roi_end, roi_color, 2)
            cv2.putText(draw_img, roi_name, (roi_start[0]+5, roi_start[1]+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

        # 3. 绘制所有检测到的区域（上料检测）
        if (ptype == PType.FEED_CHECK or ptype == PType.FEED_END_CHECK) and len(self.detected_regions) > 0:
            for i, region in enumerate(self.detected_regions):
                color = region.get("color", (0,255,0))
                
                # 绘制产品边界框（如果有旋转矩形）
                if "rotated_rect" in region and region["rotated_rect"] is not None:
                    box = cv2.boxPoints(region["rotated_rect"])
                    box = np.int32(box)
                    cv2.drawContours(draw_img, [box], 0, color, 2)
                elif "line_points" in region and len(region["line_points"]) == 2:
                    # 绘制直线（兼容旧格式）
                    (x1, y1), (x2, y2) = region["line_points"]
                    cv2.line(draw_img, (x1, y1), (x2, y2), color, 3)
                    
                    # 绘制端点
                    cv2.circle(draw_img, (x1, y1), 4, (255,255,255), -1)
                    cv2.circle(draw_img, (x2, y2), 4, (255,255,255), -1)
                
                # 绘制产品矩形框（如果有外接矩形）
                if "bounding_rect" in region:
                    x, y, w, h = region["bounding_rect"]
                    cv2.rectangle(draw_img, (x, y), (x+w, y+h), color, 2)
                
                # 绘制抓取边缘线（高亮）
                edge_y = region.get("line_y_avg", 0)
                if edge_y > 0:
                    line_x1 = region["pixel_center"][0] - self.product_width // 2
                    line_x2 = region["pixel_center"][0] + self.product_width // 2
                    cv2.line(draw_img, (line_x1, int(edge_y)), (line_x2, int(edge_y)), (0,255,255), 2)
                    cv2.putText(draw_img, "GRIP EDGE", (line_x1, int(edge_y)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                
                # 标注区域序号和层信息
                mid_x = region["pixel_center"][0]
                mid_y = region["pixel_center"][1]
                cv2.putText(draw_img, f"R{i+1}", (mid_x-15, mid_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 显示层信息
                layer = region.get("layer", 0)
                cv2.putText(draw_img, f"L{layer+1}", (mid_x-15, mid_y+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 显示匹配分数（如果是模板匹配）
                if "match_score" in region:
                    cv2.putText(draw_img, f"Score:{region['match_score']:.2f}", 
                               (mid_x-15, mid_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # 绘制中心点
                cx, cy = region["pixel_center"]
                cv2.circle(draw_img, (cx, cy), 5, color, -1)
                
                # 绘制深度信息
                depth_text = f"D:{region['avg_depth']:.0f}mm"
                cv2.putText(draw_img, depth_text, (cx+40, cy-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # 特别标注主检测结果（根据配置的索引）
            if len(self.detected_regions) > 0:
                feed_edge_index = min(getattr(self, 'feed_edge_index', 0), len(self.detected_regions)-1)
                main_region = self.detected_regions[feed_edge_index]
                
                # 高亮主检测区域
                if "rotated_rect" in main_region and main_region["rotated_rect"] is not None:
                    box = cv2.boxPoints(main_region["rotated_rect"])
                    box = np.int32(box)
                    cv2.drawContours(draw_img, [box], 0, (255,255,255), 3)
                
                if "line_points" in main_region:
                    (x1, y1), (x2, y2) = main_region["line_points"]
                    cv2.line(draw_img, (x1, y1), (x2, y2), (255,255,255), 2)
                
                cv2.putText(draw_img, "MAIN", (x1-10, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # 4. 绘制YOLO铁屑检测框（仅铁屑检测）
        if ptype == PType.IRON_CHIP_CHECK and len(self.detect_iron_chips) > 0:
            for (x1, y1, w, h, conf) in self.detect_iron_chips:
                cv2.rectangle(draw_img, (x1, y1), (x1+w, y1+h), (0,0,255), 2)
                cv2.putText(draw_img, f"Chip {conf}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # 5. 绘制下料区域的理论放置位置（仅下料检测）
        if ptype == PType.UNLOAD_CHECK:
            h, w = draw_img.shape[:2]
            roi_x_start = max(self.unload_roi_x, 0)
            roi_x_end = min(self.unload_roi_x + self.unload_roi_w, w)
            roi_y_start = max(self.unload_roi_y, 0)
            
            # 计算每层的理论坐标
            for layer_idx in range(self.unload_layer_count):
                start_x = roi_x_start
                start_y = roi_y_start
                
                for item_idx in range(self.unload_item_count_per_layer):
                    item_x = start_x
                    item_y = start_y + item_idx * (self.unload_item_height + self.unload_item_interval)
                    
                    # 绘制位置框
                    cv2.rectangle(draw_img, 
                                 (item_x, item_y),
                                 (item_x + self.unload_item_width, item_y + self.unload_item_height),
                                 (255,255,0), 1)
                    cv2.putText(draw_img, f"L{layer_idx} I{item_idx}", (item_x-20, item_y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
            
            # 绘制第一个空余位置（红色高亮）
            if exists_flag == DetectStatus.EXIST:
                target_x = int(coords[0] * self.camera_fx / coords[2] + self.camera_cx)
                target_y = int(coords[1] * self.camera_fy / coords[2] + self.camera_cy)
                cv2.circle(draw_img, (target_x, target_y), 8, (0,0,255), -1)
                cv2.putText(draw_img, "TARGET", (target_x+10, target_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # 6. 绘制图例
        legend_y = draw_img.shape[0] - 80
        cv2.putText(draw_img, "Legend:", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(draw_img, "Red: ROI", (20, legend_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        cv2.putText(draw_img, "Green/Yellow: Detected Region", (20, legend_y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        cv2.putText(draw_img, "Yellow Line: Grip Edge", (20, legend_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        return draw_img

    def depth_pseudo_color(self, depth_img):
        # 1. 深度图转【伪彩色热力图】核心步骤 (深度值归一化到0-255 + 上色)
        depth_show = depth_img.copy()
        # 归一化深度值到 0-255 (只对ROI内的有效深度做归一化，排除无效值)
        #roi_x_start = max(self.roi_x, 0)
        #roi_x_end = min(self.roi_x + self.roi_w, depth_show.shape[1])
        #roi_y_start = max(self.roi_y, 0)
        #roi_y_end = min(self.roi_y + self.roi_h, depth_show.shape[0])
        #roi_depth = depth_show[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_depth = depth_show
        # 过滤无效深度值，只计算有效深度的最大最小值
        valid_depth = roi_depth[(roi_depth != self.depth_invalid) & (roi_depth >= self.feed_depth_min) & (roi_depth <= self.feed_depth_max)]
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
def get_rgb_filename(depth_filename):
    """根据深度图文件名获取对应的RGB文件名"""
    depth_path = Path(depth_filename)
    
    # 直接替换 depth_ 为 rgb_，并修改扩展名
    rgb_name = depth_path.name.replace("depth_", "rgb_", 1)
    rgb_name = rgb_name.replace(".png", ".jpg")  # 假设RGB是jpg格式
    
    rgb_path = depth_path.parent / rgb_name
    
    # 如果jpg不存在，尝试png
    if not rgb_path.exists():
        rgb_path = depth_path.parent / rgb_name.replace(".jpg", ".png")
    
    return str(rgb_path) if rgb_path.exists() else None

from src.depthSegmentPython import DetectionStatistics

if __name__ == "__main__":
    detector = RGBDDetector()
    product_no = "M001"
    init_res = detector.init(product_no)
    if init_res["code"] != 0:
        print(f"初始化失败: {init_res['err_msg']}")
        exit(-1)
    print("初始化成功！")

    # depth_filename = "data/20260123/depth_1769134430866.png"
    
    # 默认rgb图像
    # rgb_img = cv2.imread("./rgb_image.png")
    # rgb_img = cv2.imread("data/20260123/rgb_1769134430866.jpg")

    # 获取所有PNG文件

    image_folder = "data/20260409"
    image_files = list(Path(image_folder).glob("*.png"))

    camera_img_folder = Path(__file__).parent.parent.parent / "camera_img/20260409"

    image_files = list(camera_img_folder.glob("*.png"))
    print(image_files)

    if not image_files:
        print(f"在文件夹 {image_folder} 中没有找到PNG文件")
        exit(-1)
    print(f"找到 {len(image_files)} 个PNG文件")
    
    # 创建输出文件夹
    output_folder = image_folder + "_result"
    os.makedirs(output_folder, exist_ok=True)

    # ===================== 初始化统计类 =====================
    stats = DetectionStatistics.DetectionStatistics()
    stats.set_total_files(len(image_files))

    for i, image_file in enumerate(image_files):
        if image_file.name.startswith("detect_result_horizontal_line"):
            continue
        # if i < 81:
        #     continue
        print(f"\n处理文件 {i+1}/{len(image_files)}: {image_file.name}")
        depth_filename = str(image_file)
        depth_img = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)

        # 获取对应的RGB图像
        rgb_filename = get_rgb_filename(str(depth_filename))
        if not rgb_filename:
            print(f"未找到对应的RGB图像: {depth_filename}")
            continue
        rgb_img = cv2.imread(rgb_filename)

        if rgb_img is None or depth_img is None:
            print("读取图像失败，请检查路径！")
            exit(-1)
            
        if depth_img.dtype == np.uint8:
            depth_img = depth_img.astype(np.uint16) * 20
        # elif depth_img.dtype == np.uint16:
        #     depth_img8u = (depth_img / 20).astype(np.uint8)
        #     cv2.imwrite("./depth_8u.png", depth_img8u)

        # 切换排序规则 → 联动扫描方向
        # detector.sort_rule = SortRule.SORT_BY_Y_DESC    # Y降序 → 从底部向上找水平直线
        # detector.sort_rule = SortRule.SORT_BY_Y_ASC   # Y升序 → 从顶部向下找水平直线
        
        # ptype = PType.MATERIAL_CHECK
        ptype = PType.FEED_CHECK
        # ptype = PType.UNLOAD_CHECK
        detect_res = detector.detect(ptype, rgb_img, depth_img)
        print("检测结果:\n", json.dumps(detect_res, ensure_ascii=False, indent=2))

        # 记录结果
        if detect_res["code"] == 0 and detect_res["result"]["exists"] == DetectStatus.EXIST:
            coords = detect_res["result"]["coords"]
            stats.add_success(image_file.name, coords)
            print(f"检测成功: X={coords[0]:.2f}, Y={coords[1]:.2f}, Z={coords[2]:.2f}, R={coords[3]:.2f}")
        else:
            err_msg = detect_res.get('err_msg', '检测失败')
            print(f"检测失败: {err_msg}")
            stats.add_failure(image_file.name, err_msg)

        depth_color = detector.depth_pseudo_color(depth_img)

        result_img = detector.draw_result_with_rotated_box(depth_color, detect_res)
        cv2.imshow("result-line", result_img)
        # file_name = depth_filename.rsplit('.', 1)[0] + "_result.jpg"
        file_name = os.path.join(output_folder, f"{image_file.stem}_result.jpg")
        # cv2.imwrite(file_name, result_img)

        if True: # ===================== 输出统计结果 =====================
            # stats.print_summary()
            # 保存统计结果
            # stats.save_statistics(output_folder)
            # 绘制分布图
            # stats.plot_distribution(output_folder)
            # 获取中位数和平均值
            median_coords = stats.get_median_coords()
            mean_coords = stats.get_mean_coords()
            
            if median_coords:
                print(f"中位数坐标: X={median_coords['x']:.2f}, Y={median_coords['y']:.2f}, "
                    f"Z={median_coords['z']:.2f}, R={median_coords['r']:.2f}")
                print(f"平均值坐标: X={mean_coords['x']:.2f}, Y={mean_coords['y']:.2f}, "
                    f"Z={mean_coords['z']:.2f}, R={mean_coords['r']:.2f}")

        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
