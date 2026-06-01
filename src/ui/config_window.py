# -*- coding: utf-8 -*-
import sys
import json
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QFormLayout, QLabel, QLineEdit, QPushButton,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                             QSplitter, QListWidget, QMessageBox, QComboBox, QInputDialog,
                             QTreeWidget, QTreeWidgetItem, QFrame, QDialog, QDialogButtonBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from src.utils.config_manager import CONFIG_FILE
from src.core.kinematics import ScaraKinematics

class GroupInputDialog(QDialog):
    """自定义对话框：用于同时输入分组名称和 JSON Key"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建点位分组")
        self.setMinimumWidth(350)

        # 主布局
        layout = QVBoxLayout(self)

        # 表单布局存放输入框
        form_layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如: 扫码点位")
        form_layout.addRow("显示名称:", self.name_edit)

        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("要求全英文/下划线，例如: barcode_points")
        form_layout.addRow("JSON Key:", self.key_edit)

        layout.addLayout(form_layout)

        # 确定和取消按钮
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addWidget(self.button_box)


    def get_data(self):
        """返回用户输入的数据 (display_name, json_key)"""
        return self.name_edit.text().strip(), self.key_edit.text().strip()


class ConfigEditorUI(QMainWindow):
    def __init__(self, controller_instance):
        super().__init__()
        self.setWindowTitle("SCARA 参数配置与监控中心")
        self.resize(1600, 900)

        self.controller = controller_instance  # 持有控制器的引用
        # self.plc = self.controller.plc  # 复用控制器的 PLC 连接

        self.config_data = {}
        # self.config_file = "config.json"
        self.config_file = CONFIG_FILE

        # 初始化界面
        self.init_ui()

        # 加载配置
        self.load_config_from_file()

        # === 启动 UI 刷新定时器 ===
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.refresh_realtime_display)
        # 设置刷新频率，例如 200ms (5Hz)，人眼看着流畅即可，不必太快
        self.ui_timer.start(200)

        # 【新增】垫木提示框实例
        self.wood_stick_dialog = None

        # 上料取垫木提示框
        self.controller.sig_wood_stick_alarm.connect(self.show_wood_stick_dialog)
        self.controller.sig_wood_stick_clear.connect(self.close_wood_stick_dialog)

        # 下料放垫木提示框
        self.unloading_wood_stick_dialog = None
        self.controller.sig_unloading_wood_stick_alarm.connect(self.show_unloading_wood_place_dialog)
        self.controller.sig_unloading_wood_stick_clear.connect(self.close_unloading_wood_place_dialog)



    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # === 左侧：实时监控面板 ===
        left_panel = self.create_monitor_panel()
        main_layout.addWidget(left_panel, 1)  # 权重1

        # === 右侧：配置编辑面板 ===
        right_panel = self.create_config_panel()
        main_layout.addWidget(right_panel, 4)  # 权重4

        self.btn_run_control = QPushButton("停止服务")
        self.btn_run_control.clicked.connect(self.toggle_controller)

    def toggle_controller(self):
        if self.controller.isRunning():
            self.controller.stop_service()
            self.btn_run_control.setText("启动服务")
            self.btn_run_control.setStyleSheet("background-color: green; color: white;")
        else:
            self.controller.start()
            self.btn_run_control.setText("停止服务")
            self.btn_run_control.setStyleSheet("background-color: red; color: white;")

    def create_monitor_panel(self):
        panel = QGroupBox("实时监控状态")
        layout = QFormLayout()

        # 样式设置
        style = "font-size: 16pt; font-weight: bold; color: #1976D2;"

        # --- 1. 电机角度部分 ---
        layout.addRow(QLabel("<b>[ 电机关节 ]</b>"))
        self.lbl_j1 = QLabel("0.00")
        self.lbl_j2 = QLabel("0.00")
        self.lbl_j3 = QLabel("0.00")
        self.lbl_j4 = QLabel("0.00")

        layout.addRow("轴1 角度 (°):", self.lbl_j1)
        layout.addRow("轴2 角度 (°):", self.lbl_j2)
        layout.addRow("轴3 高度 (mm):", self.lbl_j3)
        layout.addRow("轴4 角度 (°):", self.lbl_j4)

        # --- 2. 空间坐标部分 (新增) ---
        # 加一个分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addRow(line)

        layout.addRow(QLabel("<b>[ 空间坐标 ]</b>"))
        self.lbl_x = QLabel("0.00")
        self.lbl_y = QLabel("0.00")
        self.lbl_z = QLabel("0.00")
        self.lbl_r = QLabel("0.00")

        # 空一行或分割
        layout.addRow("X (mm):", self.lbl_x)
        layout.addRow("Y (mm):", self.lbl_y)
        layout.addRow("Z (mm):", self.lbl_z)
        layout.addRow("R (°):", self.lbl_r)

        # 统一应用样式
        all_labels = [
            self.lbl_j1, self.lbl_j2, self.lbl_j3, self.lbl_j4,
            self.lbl_x, self.lbl_y, self.lbl_z, self.lbl_r
        ]

        for lbl in all_labels:
            lbl.setStyleSheet(style)

        panel.setLayout(layout)
        return panel

    def create_config_panel(self):
        tabs = QTabWidget()

        # Tab 1: 全局参数
        self.tab_global = QWidget()
        self.init_global_tab()
        tabs.addTab(self.tab_global, "全局参数")

        # Tab 2: 动作流程编辑
        self.tab_process = QWidget()
        self.init_process_tab()
        # tabs.addTab(self.tab_process, "动作流程(Process)")
        tabs.addTab(self.tab_process, "动作流程配置")

        # Tab 3: 产品配置 ===
        self.tab_product = QWidget()
        self.init_product_tab()
        tabs.addTab(self.tab_product, "产品配置")

        # Tab 4: 工装夹具 ===
        self.tab_tools = QWidget()
        self.init_tools_tab()
        tabs.addTab(self.tab_tools, "工装夹具(Tools)")

        return tabs

    def init_global_tab(self):
        layout = QVBoxLayout(self.tab_global)

        # 1. PLC Config
        grp_plc = QGroupBox("PLC 配置")
        form_plc = QFormLayout()
        self.edit_ip = QLineEdit()
        self.edit_port = QLineEdit()
        form_plc.addRow("IP 地址:", self.edit_ip)
        form_plc.addRow("端口:", self.edit_port)
        grp_plc.setLayout(form_plc)

        # 2. Robot Params
        grp_robot = QGroupBox("机械臂参数")
        form_robot = QFormLayout()
        self.edit_l1 = QLineEdit()
        self.edit_l2 = QLineEdit()
        self.edit_z0 = QLineEdit()
        self.edit_nn3 = QLineEdit()
        form_robot.addRow("L1 (大臂):", self.edit_l1)
        form_robot.addRow("L2 (小臂):", self.edit_l2)
        form_robot.addRow("Z0 (基准):", self.edit_z0)
        form_robot.addRow("NN3 (系数):", self.edit_nn3)
        grp_robot.setLayout(form_robot)

        # 3. Trajectory Params
        grp_traj = QGroupBox("轨迹参数")
        form_traj = QFormLayout()
        self.edit_accel = QLineEdit()
        form_traj.addRow("加速时间 (s):", self.edit_accel)
        grp_traj.setLayout(form_traj)

        # Save Button
        btn_save = QPushButton("保存全局配置")
        btn_save.setFixedHeight(40)
        btn_save.clicked.connect(self.save_global_config)

        layout.addWidget(grp_plc)
        layout.addWidget(grp_robot)
        layout.addWidget(grp_traj)
        layout.addWidget(btn_save)
        layout.addStretch()

    def init_process_tab(self):
        layout = QHBoxLayout(self.tab_process)

        # 左侧：动作列表
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("动作列表 (Process ID/H地址)"))

        # self.list_processes = QListWidget()
        # self.list_processes.currentRowChanged.connect(self.on_process_selected)
        # left_layout.addWidget(self.list_processes)

        # 【修改】使用 QTreeWidget 替代 QListWidget 以支持多列显示
        self.tree_processes = QTreeWidget()
        self.tree_processes.setColumnCount(2)
        self.tree_processes.setHeaderLabels(["H地址 (Key)", "D地址"])
        self.tree_processes.setRootIsDecorated(False)  # 隐藏展开小箭头，使其看起来像列表

        # 设置列宽比例 (H地址窄一点，D地址宽一点，或者平分)
        self.tree_processes.setColumnWidth(0, 120)

        # 连接信号：注意信号变了，变成 currentItemChanged
        self.tree_processes.currentItemChanged.connect(self.on_process_selected)

        left_layout.addWidget(self.tree_processes)

        # 新增：动作列表的操作按钮
        proc_btn_layout = QHBoxLayout()
        self.btn_add_proc = QPushButton("新建动作")
        self.btn_del_proc = QPushButton("删除动作")
        self.btn_add_proc.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_del_proc.setStyleSheet("background-color: #F44336; color: white;")

        self.btn_add_proc.clicked.connect(self.add_process_item)
        self.btn_del_proc.clicked.connect(self.delete_process_item)

        proc_btn_layout.addWidget(self.btn_add_proc)
        proc_btn_layout.addWidget(self.btn_del_proc)
        left_layout.addLayout(proc_btn_layout)

        # 右侧：点位表格
        right_layout = QVBoxLayout()

        # 顶部信息
        form_proc_info = QFormLayout()

        self.lbl_proc_name = QLineEdit()
        self.lbl_proc_name.setPlaceholderText("例如：臂去取料位")

        self.lbl_proc_type = QLineEdit()
        self.lbl_proc_type.setPlaceholderText("例如：vision_trigger")

        self.lbl_proc_d_addr = QLineEdit()
        self.lbl_proc_d_addr.setPlaceholderText("自动计算...")
        self.lbl_proc_d_addr.setReadOnly(True)  # 设置为只读，防止用户乱填
        self.lbl_proc_d_addr.setStyleSheet("background-color: #f0f0f0; color: #555;")  # 灰色背景示意外观

        form_proc_info.addRow("动作名称:", self.lbl_proc_name)
        form_proc_info.addRow("动作类型:", self.lbl_proc_type)
        form_proc_info.addRow("PLC D地址:", self.lbl_proc_d_addr)

        right_layout.addLayout(form_proc_info)

        # ========================================================
        # 2. 【新增】分组管理工具栏 (在 Tab 上方)
        # ========================================================
        group_toolbar = QHBoxLayout()

        btn_add_group = QPushButton("+ 新建点位分组")
        btn_edit_group = QPushButton("✎ 编辑当前分组")
        btn_del_group = QPushButton("- 删除当前分组")

        # 稍微设置一下样式区分
        btn_add_group.setStyleSheet("color: #4CAF50; font-weight: bold;")
        btn_del_group.setStyleSheet("color: #F44336;")

        btn_add_group.clicked.connect(self.add_point_group)
        btn_edit_group.clicked.connect(self.edit_point_group)
        btn_del_group.clicked.connect(self.delete_point_group)

        group_toolbar.addWidget(btn_add_group)
        group_toolbar.addWidget(btn_edit_group)
        group_toolbar.addWidget(btn_del_group)
        group_toolbar.addStretch()  # 靠左对齐

        right_layout.addLayout(group_toolbar)

        right_layout.addSpacing(15)  # 15px 的留白间距，可自行调大调小

        # 创建内部 TabWidget 管理两个表格
        self.tabs_points = QTabWidget()

        self.tabs_points.setStyleSheet(
        """
            QTabBar::tab {
                background-color: #E0E0E0;  /* 未选中时的浅灰背景 */
                color: #333333;             /* 未选中时的字体颜色 */
                padding: 8px 10px;          /* 标签内边距，让标签大一点 */
                
                min-width: 155px;         /* 标签最小宽度（根据你的文字长度调整） */
                
                border-top-left-radius: 4px;/* 圆角 */
                border-top-right-radius: 4px;
                margin-right: 0px;          /* 标签之间的间距 */
                font-size: 10pt;
            }
            QTabBar::tab:hover:!selected {
                background-color: #B0BEC5;  /* 鼠标悬浮但未选中时的颜色 */
            }
            QTabBar::tab:selected {
                background-color: #2196F3;  /* 选中时的蓝色背景 */
                color: white;               /* 选中时的白色字体 */
                font-weight: bold;          /* 选中时字体加粗 */
            }
            QTabWidget::pane {
                border: 1px solid #2196F3;  /* 给下方的表格区域加一圈蓝色边框，视觉更统一 */
                top: -1px; 
            }
            """
        )

        # # 1. 普通点位
        # self.table_normal_points = self._create_points_table()
        # self.tabs_points.addTab(self.table_normal_points, "普通点位")
        #
        # # 2. 阵列搜寻点位
        # self.table_search_points = self._create_points_table()
        # self.tabs_points.addTab(self.table_search_points, "上料阵列搜寻点位")
        #
        # # 3. 翻肘安全过渡点位
        # self.table_flip_points = self._create_points_table()
        # self.tabs_points.addTab(self.table_flip_points, "上料翻肘过渡点位")
        #
        # # 4. 上料端头搜寻点位
        # self.table_layer_head_points = self._create_points_table()
        # self.tabs_points.addTab(self.table_layer_head_points, "上料端头搜寻点位")

        right_layout.addWidget(self.tabs_points)

        # 点位操作按钮
        btn_layout = QHBoxLayout()
        btn_add_pt = QPushButton("添加点位")
        btn_del_pt = QPushButton("删除选中点")
        # 读取实时坐标按钮
        btn_read_pos = QPushButton("示教(读取当前位置)")

        btn_up = QPushButton("↑ 上移")
        btn_up.setStyleSheet("color: #1976D2; font-weight: bold;")

        btn_down = QPushButton("↓ 下移")
        btn_down = QPushButton("↓ 下移")

        btn_save = QPushButton("保存当前动作")
        btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 30px;")

        btn_add_pt.clicked.connect(self.add_point_row)
        btn_del_pt.clicked.connect(self.delete_point_row)
        btn_read_pos.clicked.connect(self.teach_current_position)

        btn_up.clicked.connect(self.move_point_up)
        btn_down.clicked.connect(self.move_point_down)

        btn_save.clicked.connect(self.save_current_process)

        btn_layout.addWidget(btn_add_pt)
        btn_layout.addWidget(btn_del_pt)
        btn_layout.addWidget(btn_up)
        btn_layout.addWidget(btn_down)
        btn_layout.addWidget(btn_read_pos)
        btn_layout.addWidget(btn_save)
        right_layout.addLayout(btn_layout)

        # 设置左右比例 1:3
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 3)

    def init_product_tab(self):
        """初始化产品配置 Tab (重构版)"""
        layout = QVBoxLayout(self.tab_product)
        layout.setSpacing(20)

        # === 1. 当前生产产品 (只读展示) ===
        grp_current = QGroupBox("当前生产产品")
        layout_current = QVBoxLayout(grp_current)

        # 使用黄色大号字体显示
        self.lbl_current_product = QLabel("未设置")
        self.lbl_current_product.setAlignment(Qt.AlignCenter)
        self.lbl_current_product.setStyleSheet("""
            background-color: #333333; 
            color: #FFEB3B; 
            font-size: 24pt; 
            font-weight: bold; 
            border-radius: 5px; 
            padding: 10px;
        """)
        layout_current.addWidget(self.lbl_current_product)
        layout.addWidget(grp_current)

        # === 2. 产品型号切换 (操作区) ===
        grp_switch = QGroupBox("产品型号切换")
        layout_switch = QHBoxLayout(grp_switch)

        self.combo_product = QComboBox()
        self.combo_product.setMinimumHeight(40)
        self.combo_product.setStyleSheet("font-size: 12pt;")

        btn_switch = QPushButton("确认切换")
        btn_switch.setMinimumHeight(40)
        btn_switch.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 11pt;")
        btn_switch.clicked.connect(self.switch_product_model)  # 连接到新的切换函数

        layout_switch.addWidget(self.combo_product, 3)  # 比例 3
        layout_switch.addWidget(btn_switch, 1)  # 比例 1
        layout.addWidget(grp_switch)

        # === 3. 型号管理 (增删配置) ===
        grp_manage = QGroupBox("型号管理")
        layout_manage = QHBoxLayout(grp_manage)

        btn_add = QPushButton("添加新型号")
        btn_del = QPushButton("删除选中型号")
        # 仅仅保存列表变更，不切换型号
        btn_save_list = QPushButton("保存列表变更")

        btn_add.setStyleSheet("background-color: #2196F3; color: white;")
        btn_del.setStyleSheet("background-color: #F44336; color: white;")

        btn_add.clicked.connect(self.add_product_model)
        btn_del.clicked.connect(self.delete_product_model)
        btn_save_list.clicked.connect(self.save_product_list_only)

        layout_manage.addWidget(btn_add)
        layout_manage.addWidget(btn_del)
        layout_manage.addWidget(btn_save_list)
        layout.addWidget(grp_manage)

        layout.addStretch()

    def init_tools_tab(self):
        """初始化工装夹具配置 Tab (样式升级版)"""
        layout = QVBoxLayout(self.tab_tools)
        layout.setSpacing(20)  # 增加间距，使布局更舒展

        # === 区域 1: 当前使用夹具 (只读展示 - 样式同步产品配置) ===
        grp_current = QGroupBox("当前使用夹具")
        layout_current = QVBoxLayout(grp_current)

        self.lbl_current_tool = QLabel("未设置")
        self.lbl_current_tool.setAlignment(Qt.AlignCenter)
        # 黑色背景，黄色大字
        self.lbl_current_tool.setStyleSheet("""
            background-color: #333333; 
            color: #FFEB3B; 
            font-size: 24pt; 
            font-weight: bold; 
            border-radius: 5px; 
            padding: 10px;
        """)
        layout_current.addWidget(self.lbl_current_tool)
        layout.addWidget(grp_current)

        # === 区域 2: 夹具切换 (操作区 - 样式同步产品配置) ===
        grp_switch = QGroupBox("夹具型号切换")
        layout_switch = QHBoxLayout(grp_switch)

        self.combo_tools = QComboBox()
        self.combo_tools.setMinimumHeight(40)  # 加高
        self.combo_tools.setStyleSheet("font-size: 12pt;")

        btn_switch = QPushButton("确认切换")
        btn_switch.setMinimumHeight(40)  # 加高
        # 绿色按钮
        btn_switch.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 11pt;")
        btn_switch.clicked.connect(self.switch_tool_model)

        layout_switch.addWidget(self.combo_tools, 3)  # 下拉框占 3 份
        layout_switch.addWidget(btn_switch, 1)  # 按钮占 1 份
        layout.addWidget(grp_switch)

        # === 区域 3: 详细配置管理 (左右分栏) ===
        # 这里不需要 GroupBox 包裹，直接作为下半部分
        bottom_layout = QHBoxLayout()

        # --- 左侧：列表管理 ---
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("夹具型号列表"))

        self.list_tools = QListWidget()
        self.list_tools.currentRowChanged.connect(self.on_tool_selected)
        left_layout.addWidget(self.list_tools)

        # 增删按钮
        btn_box = QHBoxLayout()
        btn_add = QPushButton("新建夹具")
        btn_del = QPushButton("删除选中")
        btn_add.setStyleSheet("background-color: #2196F3; color: white;")
        btn_del.setStyleSheet("background-color: #F44336; color: white;")

        btn_add.clicked.connect(self.add_tool_model)
        btn_del.clicked.connect(self.delete_tool_model)

        btn_box.addWidget(btn_add)
        btn_box.addWidget(btn_del)
        left_layout.addLayout(btn_box)

        # --- 右侧：参数编辑 ---
        right_grp = QGroupBox("参数详情")
        right_layout = QVBoxLayout(right_grp)

        # 1. 基础信息
        form_base = QFormLayout()
        self.tool_name_edit = QLineEdit()
        self.tool_name_edit.setReadOnly(True)  # 禁止直接改名，防止ID错乱
        self.tool_name_edit.setPlaceholderText("在左侧列表选择...")
        self.tool_desc_edit = QLineEdit()
        form_base.addRow("型号名称:", self.tool_name_edit)
        form_base.addRow("描述备注:", self.tool_desc_edit)
        right_layout.addLayout(form_base)

        # 2. 相机配置
        grp_cam = QGroupBox("相机参数 (Camera)")
        form_cam = QFormLayout(grp_cam)
        self.cam_off_x = QLineEdit()
        self.cam_off_y = QLineEdit()
        self.cam_rot = QLineEdit()
        form_cam.addRow("Offset X:", self.cam_off_x)
        form_cam.addRow("Offset Y:", self.cam_off_y)
        form_cam.addRow("Rotation:", self.cam_rot)
        right_layout.addWidget(grp_cam)

        # 3. 夹爪配置
        grp_grip = QGroupBox("夹爪参数 (Main Gripper)")
        form_grip = QFormLayout(grp_grip)
        self.grip_off_x = QLineEdit()
        self.grip_off_y = QLineEdit()
        self.grip_z_diff = QLineEdit()
        form_grip.addRow("Offset X:", self.grip_off_x)
        form_grip.addRow("Offset Y:", self.grip_off_y)
        form_grip.addRow("Z Diff:", self.grip_z_diff)
        right_layout.addWidget(grp_grip)

        # 保存按钮
        btn_save_params = QPushButton("保存参数修改")
        btn_save_params.setMinimumHeight(35)
        btn_save_params.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        btn_save_params.clicked.connect(self.save_current_tool_params)
        right_layout.addWidget(btn_save_params)
        right_layout.addStretch()

        # 组合左右布局 (左1 : 右2)
        bottom_layout.addLayout(left_layout, 1)
        bottom_layout.addWidget(right_grp, 2)

        layout.addLayout(bottom_layout)

    # === 逻辑处理部分 ===
    def _create_points_table(self):
        """创建一个标准的点位表格控件"""
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(["点名称", "X", "Y", "Z", "R (te)", "Photo", "姿态"])
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        return table

    def _get_current_active_table(self):
        """获取当前正在显示的那个表格对象"""
        return self.tabs_points.currentWidget()

    def _populate_points_table(self, table, points_data):
        """向指定的表格中填充数据"""
        table.setRowCount(len(points_data))
        table.setSortingEnabled(False)

        for i, pt in enumerate(points_data):
            table.setItem(i, 0, QTableWidgetItem(str(pt.get('name', ''))))
            coords = pt.get('coords', [0, 0, 0, 0])
            for j in range(4):
                val = f"{coords[j]:.2f}"
                table.setItem(i, j + 1, QTableWidgetItem(val))

            # 设置photo下拉框
            photo_val = pt.get('photo', 0)
            self._set_row_photo_combo(table, i, current_val=photo_val)

            # 设置 姿态 下拉框
            current_config = pt.get('config', 'elbow_up')
            self._set_row_elbow_combo(table, i, current_val=current_config)

        table.setSortingEnabled(False)

    def _extract_points_from_table(self, table):
        """从指定的表格中提取数据生成列表"""
        extracted_points = []
        rows = table.rowCount()
        for i in range(rows):
            name = table.item(i, 0).text()
            x = float(table.item(i, 1).text())
            y = float(table.item(i, 2).text())
            z = float(table.item(i, 3).text())
            r = float(table.item(i, 4).text())

            # 读取 Photo 下拉框的值
            combo_photo = table.cellWidget(i, 5)
            photo = combo_photo.currentIndex() if combo_photo else 0

            # 读取 姿态 下拉框的值
            combo = table.cellWidget(i, 6)
            config_val = combo.currentText() if combo else "elbow_up"

            extracted_points.append({
                "name": name,
                "coords": [x, y, z, r],
                "photo": photo,
                "config": config_val
            })
        return extracted_points

    def _set_row_elbow_combo(self, table, row, current_val="elbow_up"):
        """辅助函数：给指定行设置姿态下拉框"""
        combo_config = QComboBox()
        combo_config.addItems(["elbow_up", "elbow_down"])

        idx = combo_config.findText(current_val)
        if idx >= 0:
            combo_config.setCurrentIndex(idx)
        else:
            combo_config.setCurrentIndex(0)

        table.setCellWidget(row, 6, combo_config)

    def _set_row_photo_combo(self, table, row, current_val=0):
        """辅助函数：给指定行设置拍照动作下拉框"""
        combo_photo = QComboBox()
        # 注意：这里的选项顺序非常重要，它们的 index 刚好对应 0, 1, 2, 3
        combo_photo.addItems(["无", "深度", "CCD", "激光"])

        # 确保传入的值在安全范围内 (0~3)
        try:
            current_val = int(current_val)
            if 0 <= current_val <= 3:
                combo_photo.setCurrentIndex(current_val)
            else:
                combo_photo.setCurrentIndex(0)
        except (ValueError, TypeError):
            combo_photo.setCurrentIndex(0)

        table.setCellWidget(row, 5, combo_photo)

    def swap_table_rows(self, table, row1, row2):
        """核心逻辑：交换两行的数据"""
        # 交换普通单元格 (0-4列: Name, X, Y, Z, R)
        for col in range(5):
            item1 = table.takeItem(row1, col)
            item2 = table.takeItem(row2, col)
            table.setItem(row2, col, item1)
            table.setItem(row1, col, item2)

        # 交换 Photo 下拉框 (第5列)
        combo_photo1 = table.cellWidget(row1, 5)
        combo_photo2 = table.cellWidget(row2, 5)

        val_photo1 = combo_photo1.currentIndex() if combo_photo1 else 0
        val_photo2 = combo_photo2.currentIndex() if combo_photo2 else 0

        self._set_row_photo_combo(table, row2, current_val=val_photo1)
        self._set_row_photo_combo(table, row1, current_val=val_photo2)

        # 交换 Config/Elbow 下拉框 (第6列)
        combo_config1 = table.cellWidget(row1, 6)
        combo_config2 = table.cellWidget(row2, 6)

        val_config1 = combo_config1.currentText() if combo_config1 else "elbow_up"
        val_config2 = combo_config2.currentText() if combo_config2 else "elbow_up"

        self._set_row_elbow_combo(table, row2, current_val=val_config1)
        self._set_row_elbow_combo(table, row1, current_val=val_config2)

    def move_point_up(self):
        """上移当前行"""
        # row = self.table_points.currentRow()
        # # 如果没选中，或者已经是第一行，无法上移
        # if row <= 0:
        #     return
        #
        # # 交换当前行与上一行
        # self.swap_table_rows(row, row - 1)
        # # 保持选中状态跟随移动
        # self.table_points.setCurrentCell(row - 1, 0)

        active_table = self._get_current_active_table()
        row = active_table.currentRow()
        if row <= 0: return
        self.swap_table_rows(active_table, row, row - 1)
        active_table.setCurrentCell(row - 1, 0)

    def move_point_down(self):
        """下移当前行"""
        # row = self.table_points.currentRow()
        # count = self.table_points.rowCount()
        #
        # # 如果没选中，或者是最后一行，无法下移
        # if row < 0 or row >= count - 1:
        #     return
        #
        # # 交换当前行与下一行
        # self.swap_table_rows(row, row + 1)
        #
        # # 保持选中状态跟随移动
        # self.table_points.setCurrentCell(row + 1, 0)

        active_table = self._get_current_active_table()
        row = active_table.currentRow()
        count = active_table.rowCount()
        if row < 0 or row >= count - 1: return
        self.swap_table_rows(active_table, row, row + 1)
        active_table.setCurrentCell(row + 1, 0)

    def map_modbus_address(self, full_address):
        """
        H地址 转 D地址 计算逻辑
        """
        # 如果能获取到 controller 的逻辑，优先用 controller 的
        if self.controller:
            try:
                return self.controller.plc.map_modbus_address(full_address)
            except Exception as e:
                pass

        # 本地兜底逻辑 (防止未连接PLC时UI报错)
        if 0x40000 <= full_address <= 0x4FFFF:
            return full_address - 0x40000
        return full_address

    def load_config_from_file(self):
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)

            # 填充全局参数
            plc = self.config_data.get('plc_config', {})
            self.edit_ip.setText(plc.get('ip', ''))
            self.edit_port.setText(str(plc.get('port', '')))

            robot = self.config_data.get('robot_params', {})
            self.edit_l1.setText(str(robot.get('l1', '')))
            self.edit_l2.setText(str(robot.get('l2', '')))
            self.edit_z0.setText(str(robot.get('z0', '')))
            self.edit_nn3.setText(str(robot.get('nn3', '')))

            traj = self.config_data.get('trajectory_params', {})
            self.edit_accel.setText(str(traj.get('acceleration_time', '')))

            # === 填充动作列表 ===
            # self.list_processes.clear()
            # processes = self.config_data.get('processes', {})
            # for pid in processes.keys():
            #     self.list_processes.addItem(pid)

            self.tree_processes.clear()
            processes = self.config_data.get('processes', {})

            # 排序
            try:
                sorted_keys = sorted(processes.keys(), key=lambda x: int(x, 16))
            except:
                sorted_keys = sorted(processes.keys())

            for pid in sorted_keys:
                # 1. 获取/计算 D 地址
                proc_data = processes[pid]
                d_addr = proc_data.get('d_addr')

                if d_addr is None:
                    # 如果配置里没存，实时计算
                    try:
                        h_val = int(pid, 16)
                        d_addr = self.map_modbus_address(h_val)
                    except:
                        d_addr = "-"

                # 2. 创建 TreeItem
                # 第一列设为 pid (H地址)，第二列设为 d_addr
                item = QTreeWidgetItem([str(pid), str(d_addr)])
                self.tree_processes.addTopLevelItem(item)

            # === 加载产品配置 ===
            prod_cfg = self.config_data.get('product_config', {})
            model_list = prod_cfg.get('model_list', ["PN123456", "PN14535"])
            current_model = prod_cfg.get('current_model', "Unknown")

            # 1. 更新顶部大标签
            self.lbl_current_product.setText(current_model)

            # 2. 更新下拉框
            self.combo_product.clear()
            self.combo_product.addItems(model_list)

            # 下拉框默认选中当前的，方便用户确认
            idx = self.combo_product.findText(current_model)
            if idx >= 0:
                self.combo_product.setCurrentIndex(idx)

            # === 加载工装夹具配置 ===
            tools_cfg = self.config_data.get('tools', {})
            current_tool = tools_cfg.get('current_model', "Unknown")
            models = tools_cfg.get('models', [])

            # 1. 更新顶部状态
            self.lbl_current_tool.setText(current_tool)

            # 2. 更新切换下拉框
            self.combo_tools.clear()
            self.list_tools.clear()

            for m in models:
                name = m.get('name', 'Unnamed')
                self.combo_tools.addItem(name)
                self.list_tools.addItem(name)

            # 选中当前
            idx = self.combo_tools.findText(current_tool)
            if idx >= 0: self.combo_tools.setCurrentIndex(idx)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载配置文件失败: {e}")

    def on_process_selected(self, current_item, previous_item):
        if not current_item:
            # 清空所有
            self.lbl_proc_name.clear()
            self.lbl_proc_type.clear()
            self.lbl_proc_d_addr.clear()

            # self.table_normal_points.setRowCount(0)
            # self.table_search_points.setRowCount(0)
            # self.table_flip_points.setRowCount(0)
            # self.table_layer_head_points.setRowCount(0)

            self.tabs_points.clear()  # 清空所有 Tab
            return

        pid = current_item.text(0)
        process_data = self.config_data['processes'].get(pid, {})

        self.lbl_proc_name.setText(process_data.get('name', ''))
        self.lbl_proc_type.setText(process_data.get('type', ''))

        # === 【新增】D 地址处理逻辑 ===
        # 1. 尝试从 JSON 获取
        d_addr = process_data.get('d_addr')

        # 2. 如果 JSON 里没有，或者为了保证准确性，实时计算一遍
        if d_addr is None:
            try:
                # 将 16进制字符串转为 int
                h_addr_int = int(pid, 16)
                # 计算 D 地址
                d_addr = self.map_modbus_address(h_addr_int)
            except ValueError:
                d_addr = "Error"

        self.lbl_proc_d_addr.setText(str(d_addr))

        # # 【核心修改】分别填充两个表格
        # normal_points = process_data.get('points', [])
        # search_points = process_data.get('search_points', [])
        # flip_via_points = process_data.get('flip_via_points', [])  # 新增读取
        # layer_head_points = process_data.get('layer_head_points', [])
        #
        # self._populate_points_table(self.table_normal_points, normal_points)
        # self._populate_points_table(self.table_search_points, search_points)
        # self._populate_points_table(self.table_flip_points, flip_via_points)  # 新增填充
        # self._populate_points_table(self.table_layer_head_points, layer_head_points)

        # ========================================================
        # 动态渲染 Tab 分组
        # ========================================================
        self.tabs_points.clear()  # 切换动作时，先清空旧的 Tab

        # 1. 获取或初始化分组元数据
        # 默认的 4 个经典分组，保证向下兼容旧配置文件
        default_groups_meta = {
            "points": "普通点位"
            # "search_points": "阵列搜寻点位",
            # "flip_via_points": "翻肘过渡点位",
            # "layer_head_points": "端头搜寻点位"
        }

        groups_meta = process_data.get("groups_meta")
        if not groups_meta:
            groups_meta = default_groups_meta

        # 2. 遍历元数据，动态创建 Tab 和 表格
        for json_key, display_name in groups_meta.items():
            # 创建标准表格
            table = self._create_points_table()

            # 【魔法属性】给 table 对象强行绑定这两个属性，保存时非常有用！
            table.json_key = json_key
            table.display_name = display_name

            # 从 JSON 中提取该 key 对应的点位数组 (没有则为空列表)
            points_data = process_data.get(json_key, [])

            # 填充表格数据 (复用你现有的辅助函数)
            self._populate_points_table(table, points_data)

            # 添加到 TabWidget，标签文字为 "显示名 (key)"
            tab_title = f"{display_name}"
            self.tabs_points.addTab(table, tab_title)

    def add_process_item(self):
        """新增一个动作流程"""
        # 1. 弹出对话框输入 ID
        text, ok = QInputDialog.getText(self, "新建动作", "请输入动作地址 (例如 0x40090):")

        if ok and text:
            text = text.strip().upper()
            # 简单校验
            if text in self.config_data.get('processes', {}):
                QMessageBox.warning(self, "错误", "该动作地址已存在！")
                return

            # 计算 D 地址
            try:
                h_val = int(text, 16)
                d_val = self.map_modbus_address(h_val)
            except:
                d_val = 0

            # 2. 初始化数据结构
            # new_process = {
            #     "name": "新建动作流程",
            #     "type": "standard",
            #     "points": []
            # }
            new_process = {
                "name": "新建动作流程",
                "type": "standard_move",
                "d_addr": d_val,  # 默认存入
                "points": [],
                "groups_meta": {
                    "points": "普通点位"
                }
            }

            # 3. 更新内存数据
            if 'processes' not in self.config_data:
                self.config_data['processes'] = {}
            self.config_data['processes'][text] = new_process

            # 4. 更新 UI 列表
            # self.list_processes.addItem(text)
            # # 选中新添加的项
            # self.list_processes.setCurrentRow(self.list_processes.count() - 1)

            # 更新 UI (TreeWidget)
            item = QTreeWidgetItem([str(text), str(d_val)])
            self.tree_processes.addTopLevelItem(item)

            # 选中新项
            self.tree_processes.setCurrentItem(item)

            # 5. 自动保存（可选，或者让用户点保存按钮）
            self._write_to_file()

    def delete_process_item(self):
        """删除当前选中的动作"""
        # row = self.list_processes.currentRow()
        # if row < 0:
        #     QMessageBox.warning(self, "提示", "请先选择要删除的动作")
        #     return
        #
        # pid = self.list_processes.item(row).text()

        current_item = self.tree_processes.currentItem()
        if not current_item:
            QMessageBox.warning(self, "提示", "请先选择要删除的动作")
            return

        # 获取 Key (第0列)
        pid = current_item.text(0)

        # 二次确认
        reply = QMessageBox.question(self, "确认删除",
                                     f"确定要永久删除动作 [{pid}] 及其所有点位吗？",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 1. 从内存移除
            if pid in self.config_data.get('processes', {}):
                del self.config_data['processes'][pid]

            # 2. 从 UI 移除
            # self.list_processes.takeItem(row)

            # 需要先获取当前项的索引
            index = self.tree_processes.indexOfTopLevelItem(current_item)
            self.tree_processes.takeTopLevelItem(index)

            # 3. 自动保存
            self._write_to_file()

            # 4. 通知后台刷新
            if self.controller:
                self.controller.reload_config()

    # def teach_current_position(self):
    #     """示教功能：读取当前电机坐标并填入表格"""
    #     if not self.controller:
    #         return
    #
    #     # 从 Controller 获取实时缓存 [j1, j2, j3, j4]
    #     # 注意：这里拿到的是关节角度/高度，需要根据你的业务需求决定填什么
    #     # 如果你的 points 存的是 [x, y, z, r] (笛卡尔)，你需要在这里调用正运动学转一下
    #     # 假设 Controller 有 get_realtime_point 返回的是 {coords: [x,y,z,r]}
    #
    #     try:
    #         real_point = self.controller.get_realtime_point()
    #         if not real_point:
    #             QMessageBox.warning(self, "错误", "无法获取当前机械臂坐标")
    #             return
    #
    #         x, y, z, r = real_point['coords']
    #         # 获取实时的姿态 (elbow_up/down)
    #         current_elbow = real_point.get('config', 'elbow_up')
    #
    #         # 添加新行
    #         # row = self.table_points.rowCount()
    #         # self.table_points.insertRow(row)
    #
    #         current_row = self.table_points.currentRow()
    #         if current_row >= 0:
    #             insert_idx = current_row + 1
    #         else:
    #             insert_idx = self.table_points.rowCount()
    #
    #         self.table_points.insertRow(insert_idx)
    #
    #         self.table_points.setItem(insert_idx, 0, QTableWidgetItem(f"Teach_P{insert_idx + 1}"))
    #         self.table_points.setItem(insert_idx, 1, QTableWidgetItem(f"{x:.2f}"))
    #         self.table_points.setItem(insert_idx, 2, QTableWidgetItem(f"{y:.2f}"))
    #         self.table_points.setItem(insert_idx, 3, QTableWidgetItem(f"{z:.2f}"))
    #         self.table_points.setItem(insert_idx, 4, QTableWidgetItem(f"{r:.2f}"))
    #         self.table_points.setItem(insert_idx, 5, QTableWidgetItem("0"))
    #
    #         self._set_row_elbow_combo(insert_idx, current_val=current_elbow)
    #         # 自动选中新行
    #         self.table_points.setCurrentCell(insert_idx, 0)
    #
    #     except Exception as e:
    #         QMessageBox.warning(self, "异常", f"示教失败: {e}")

    def teach_current_position(self):
        if not self.controller: return
        try:
            real_point = self.controller.get_realtime_point()
            if not real_point:
                QMessageBox.warning(self, "错误", "无法获取当前机械臂坐标")
                return

            x, y, z, r = real_point['coords']
            current_elbow = real_point.get('config', 'elbow_up')

            active_table = self._get_current_active_table()
            current_row = active_table.currentRow()
            insert_idx = current_row + 1 if current_row >= 0 else active_table.rowCount()

            active_table.insertRow(insert_idx)

            active_table.setItem(insert_idx, 0, QTableWidgetItem(f"Teach_P{insert_idx + 1}"))
            active_table.setItem(insert_idx, 1, QTableWidgetItem(f"{x:.2f}"))
            active_table.setItem(insert_idx, 2, QTableWidgetItem(f"{y:.2f}"))
            active_table.setItem(insert_idx, 3, QTableWidgetItem(f"{z:.2f}"))
            active_table.setItem(insert_idx, 4, QTableWidgetItem(f"{r:.2f}"))
            # active_table.setItem(insert_idx, 5, QTableWidgetItem("0"))
            self._set_row_photo_combo(active_table, insert_idx, current_val=0)
            self._set_row_elbow_combo(active_table, insert_idx, current_val=current_elbow)

            active_table.setCurrentCell(insert_idx, 0)

        except Exception as e:
            QMessageBox.warning(self, "异常", f"示教失败: {e}")

    def teach_flip_point(self):
        """专属示教功能：将机械臂当前实时 X,Y 坐标填入安全过渡点"""
        if not self.controller:
            return

        try:
            real_point = self.controller.get_realtime_point()
            if not real_point:
                QMessageBox.warning(self, "错误", "无法获取当前机械臂坐标")
                return

            x, y, z, r = real_point['coords']

            # 只填入 XY，因为 Z 轴是代码动态获取同层高度的
            self.edit_flip_x.setText(f"{x:.2f}")
            self.edit_flip_y.setText(f"{y:.2f}")
            self.edit_flip_z.setText(f"{z:.2f}")
            self.edit_flip_r.setText(f"{r:.2f}")

        except Exception as e:
            QMessageBox.warning(self, "异常", f"示教安全点失败: {e}")

    # def add_point_row(self):
    #     current_row = self.table_points.currentRow()
    #     if current_row >= 0:
    #         # 如果有选中行，插在选中行的下一行
    #         insert_idx = current_row + 1
    #     else:
    #         # 如果没选中，插在表格末尾
    #         insert_idx = self.table_points.rowCount()
    #
    #     # row = self.table_points.rowCount()
    #     self.table_points.insertRow(insert_idx)
    #     # 设置默认值
    #     self.table_points.setItem(insert_idx, 0, QTableWidgetItem(f"P{insert_idx + 1}"))
    #     for j in range(1, 6):
    #         self.table_points.setItem(insert_idx, j, QTableWidgetItem("0"))
    #
    #     # combo_config = QComboBox()
    #     # combo_config.addItems(["elbow_up", "elbow_down"])
    #     # self.table_points.setCellWidget(insert_idx, 6, combo_config)
    #
    #     self._set_row_elbow_combo(insert_idx)
    #
    #     # 自动选中新添加的行
    #     self.table_points.setCurrentCell(insert_idx, 0)
    #
    # def delete_point_row(self):
    #     current_row = self.table_points.currentRow()
    #     if current_row >= 0:
    #         self.table_points.removeRow(current_row)

    # === 动态分组管理槽函数 ===
    # =======================================================
    # 【新增】UI 槽函数
    # =======================================================
    def show_wood_stick_dialog(self, layer_idx):
        """显示非阻塞警告弹窗"""
        if not self.wood_stick_dialog:
            self.wood_stick_dialog = QDialog(self)
            self.wood_stick_dialog.setWindowTitle("⚠️ 人工干预请求")
            # 窗口置顶，且非模态（不阻塞后面界面的点击，虽然通常这时候也没人点）
            self.wood_stick_dialog.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
            self.wood_stick_dialog.setModal(False)

            layout = QVBoxLayout()
            self.lbl_wood_msg = QLabel("")
            self.lbl_wood_msg.setStyleSheet("""
                font-size: 28pt; 
                color: white; 
                background-color: #F44336; 
                font-weight: bold; 
                padding: 30px;
                border-radius: 10px;
            """)
            self.lbl_wood_msg.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.lbl_wood_msg)
            self.wood_stick_dialog.setLayout(layout)

        # 更新提示文本
        self.lbl_wood_msg.setText(
            # f"上料架上层物料已抓空！\n\n请取走第 {layer_idx + 1} 层的垫木！\n\n(完成后请按下机台【复位】按钮)"
            f"上料架上层物料已抓空！\n\n请取走垫木！\n\n(完成后请按下机台【复位】按钮)"
        )
        self.wood_stick_dialog.show()

    def close_wood_stick_dialog(self):
        """关闭警告弹窗"""
        if self.wood_stick_dialog and self.wood_stick_dialog.isVisible():
            self.wood_stick_dialog.hide()

    def show_unloading_wood_place_dialog(self):
        """显示非阻塞警告弹窗"""
        if not self.unloading_wood_stick_dialog:
            self.unloading_wood_stick_dialog = QDialog(self)
            self.unloading_wood_stick_dialog.setWindowTitle("⚠️ 人工干预请求")
            # 窗口置顶，且非模态（不阻塞后面界面的点击，虽然通常这时候也没人点）
            self.unloading_wood_stick_dialog.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
            self.unloading_wood_stick_dialog.setModal(False)

            layout = QVBoxLayout()
            self.lbl_wood_msg = QLabel("")
            self.lbl_wood_msg.setStyleSheet("""
                font-size: 28pt; 
                color: white; 
                background-color: #F44336; 
                font-weight: bold; 
                padding: 30px;
                border-radius: 10px;
            """)
            self.lbl_wood_msg.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.lbl_wood_msg)
            self.unloading_wood_stick_dialog.setLayout(layout)

        self.lbl_wood_msg.setText(f"下料架放置垫木！\n\n(完成后请按下机台【复位】按钮)")
        self.unloading_wood_stick_dialog.show()

    def close_unloading_wood_place_dialog(self):
        """关闭警告弹窗"""
        if self.unloading_wood_stick_dialog and self.unloading_wood_stick_dialog.isVisible():
            self.unloading_wood_stick_dialog.hide()

    def add_point_group(self):
        """新建点位分组"""
        if not self.tree_processes.currentItem():
            QMessageBox.warning(self, "提示", "请先在左侧选择一个动作！")
            return

        # 1. 弹出自定义对话框
        dialog = GroupInputDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # 2. 获取用户输入的两个值
            display_name, json_key = dialog.get_data()

            # 空值校验
            if not display_name or not json_key:
                QMessageBox.warning(self, "警告", "显示名称和 JSON Key 均不能为空！")
                return

            # 防重校验：检查现有的 tab 里有没有这个 key
            for i in range(self.tabs_points.count()):
                if getattr(self.tabs_points.widget(i), "json_key", "") == json_key:
                    QMessageBox.warning(self, "错误", f"Key '{json_key}' 已存在！请使用其他英文名。")
                    return

            # 创建新表格
            table = self._create_points_table()

            # 绑定属性
            table.json_key = json_key
            table.display_name = display_name

            # 添加到 UI (不自动保存，等用户点大保存按钮才写入文件)
            tab_title = f"{display_name}"
            self.tabs_points.addTab(table, tab_title)

            # 跳转到新 Tab
            self.tabs_points.setCurrentIndex(self.tabs_points.count() - 1)

    def edit_point_group(self):
        """编辑当前分组信息 (只允许改显示名称，禁止改Key防错)"""
        current_idx = self.tabs_points.currentIndex()
        if current_idx < 0: return

        table = self.tabs_points.widget(current_idx)
        old_name = getattr(table, "display_name", "")
        json_key = getattr(table, "json_key", "")

        new_name, ok = QInputDialog.getText(self, "编辑分组名称", f"Key: {json_key}\n请输入新的显示名称:",
                                            text=old_name)
        if ok and new_name.strip():
            new_name = new_name.strip()
            # 更新属性
            table.display_name = new_name
            # 更新 Tab 文字
            self.tabs_points.setTabText(current_idx, f"{new_name}")

    def delete_point_group(self):
        """删除当前分组"""
        current_idx = self.tabs_points.currentIndex()
        if current_idx < 0: return

        table = self.tabs_points.widget(current_idx)
        display_name = getattr(table, "display_name", "")
        json_key = getattr(table, "json_key", "")

        if json_key == "points":
            QMessageBox.warning(
                self,
                "禁止删除",
                f"无法删除分组【{display_name}】！\n\n该分组为默认分组"
            )
            return

        if table.rowCount() > 0:
            QMessageBox.warning(
                self,
                "禁止删除",
                f"无法删除分组【{display_name}】！\n\n该分组下仍有 {table.rowCount()} 个点位数据。\n请先手动删除该分组下的所有点位，然后再尝试删除此分组。"
            )
            return

        reply = QMessageBox.warning(
            self, "危险操作确认",
            f"确定要删除分组【{display_name} ({json_key})】吗？\n\n警告：该分组下的所有点位数据将被丢弃！\n(点击右下角[保存当前动作]后真正生效)",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 只在 UI 上移除。只要用户最后点了保存，这个组就不会被提取存入 JSON，达到了删除的目的
            self.tabs_points.removeTab(current_idx)
            # 注意：旧的 JSON 里那个 Key 对应的数据还在，如果不去专门清理 dict，它会变成冗余数据
            # 最干净的做法是在 save 时，用新的 key 覆盖，旧的 key (不在 tabs_points 里的) 最好 del 掉。
            # 为了简化，我们上面写的 save_current_process 采用的是“提取现有Tab重写”的逻辑。

    def add_point_row(self):
        active_table = self._get_current_active_table()
        current_row = active_table.currentRow()

        insert_idx = current_row + 1 if current_row >= 0 else active_table.rowCount()

        active_table.insertRow(insert_idx)
        active_table.setItem(insert_idx, 0, QTableWidgetItem(f"P{insert_idx + 1}"))

        # 循环添加x, y, z, r
        for j in range(1, 5):
            active_table.setItem(insert_idx, j, QTableWidgetItem("0"))

        # 初始化 Photo 和 Config 两个下拉框
        self._set_row_photo_combo(active_table, insert_idx, current_val=0)  # 默认 0 (无)
        self._set_row_elbow_combo(active_table, insert_idx)

        active_table.setCurrentCell(insert_idx, 0)

    def delete_point_row(self):
        active_table = self._get_current_active_table()
        current_row = active_table.currentRow()
        if current_row >= 0:
            active_table.removeRow(current_row)

    def save_global_config(self):
        """保存 Tab 1 的数据到 json"""
        try:
            self.config_data['plc_config']['ip'] = self.edit_ip.text()
            self.config_data['plc_config']['port'] = int(self.edit_port.text())

            self.config_data['robot_params']['l1'] = float(self.edit_l1.text())
            self.config_data['robot_params']['l2'] = float(self.edit_l2.text())
            self.config_data['robot_params']['z0'] = float(self.edit_z0.text())
            self.config_data['robot_params']['nn3'] = float(self.edit_nn3.text())

            self.config_data['trajectory_params']['acceleration_time'] = float(self.edit_accel.text())

            self._write_to_file()

            # 【关键】保存后，通知后台线程重载配置
            if self.controller:
                self.controller.reload_config()

            QMessageBox.information(self, "成功", "全局配置已保存，并已通知后台生效")
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数字")

    def save_current_process(self):
        """保存 Tab 2 的表格数据到 json"""

        robot_params = self.config_data.get('robot_params', {})
        l1 = robot_params.get('l1', 0)
        l2 = robot_params.get('l2', 0)
        z0 = robot_params.get('z0', 0)
        nn3 = robot_params.get('nn3', 0)


        current_item = self.tree_processes.currentItem()
        if not current_item:
            return

        pid = current_item.text(0)

        try:
            # # 【核心修改】分别从两个表格提取数据
            # new_points = self._extract_points_from_table(self.table_normal_points)
            # new_search_points = self._extract_points_from_table(self.table_search_points)
            # new_flip_points = self._extract_points_from_table(self.table_flip_points)  # 新增提取
            # new_layer_head_points = self._extract_points_from_table(self.table_layer_head_points)

            # 更新内存数据
            if pid not in self.config_data['processes']:
                self.config_data['processes'][pid] = {}

            self.config_data['processes'][pid]['name'] = self.lbl_proc_name.text()
            self.config_data['processes'][pid]['type'] = self.lbl_proc_type.text()

            try:
                d_addr_val = int(self.lbl_proc_d_addr.text())
                self.config_data['processes'][pid]['d_addr'] = d_addr_val
            except ValueError:
                pass

            # # 保存表格数据
            # self.config_data['processes'][pid]['points'] = new_points
            # self.config_data['processes'][pid]['search_points'] = new_search_points
            # self.config_data['processes'][pid]['flip_via_points'] = new_flip_points  # 新增保存
            # self.config_data['processes'][pid]['layer_head_points'] = new_layer_head_points

            # ========================================================
            # 【核心逻辑】遍历所有 Tab，动态提取数据并重组 JSON
            # ========================================================
            new_groups_meta = {}
            current_keys_in_tabs = []

            # 遍历当前 TabWidget 中的所有页面 (即所有的 table)
            for i in range(self.tabs_points.count()):
                table = self.tabs_points.widget(i)

                # 提取我们在 on_process_selected 时绑定的魔法属性
                json_key = getattr(table, "json_key", f"custom_group_{i}")
                display_name = getattr(table, "display_name", f"自定义组 {i}")

                # 记录到新的 meta 字典中
                new_groups_meta[json_key] = display_name
                current_keys_in_tabs.append(json_key)

                # 提取表格里的实际点位数据 (复用你现有的辅助函数)
                points_list = self._extract_points_from_table(table)

                for pt in points_list:
                    coords = pt.get('coords', [0, 0, 0, 0])
                    config_type = pt.get('config', 'elbow_up')
                    ik = ScaraKinematics.inverse_kinematics_v2(
                        coords[0], coords[1], coords[2], coords[3],
                        l1, l2, z0, nn3,
                        config_type=config_type
                    )
                    if ik:
                        pt['joints'] = [ik['the1'], ik['the2'], ik['the3'], ik['th4']]
                    else:
                        pt['joints'] = []  # 逆解失败，标记为空
                # 将点位数组直接存入 config_data 对应的 key 下
                self.config_data['processes'][pid][json_key] = points_list

            # 【新增：清理被用户在界面上删除的旧分组数组】
            # 读取旧的 meta
            old_meta = self.config_data['processes'][pid].get('groups_meta', {})
            for old_key in old_meta.keys():
                # 如果这个旧 key 不在当前界面上的 tab 列表里，说明被删了
                if old_key not in current_keys_in_tabs:
                    # 从 json 字典中移除它！
                    if old_key in self.config_data['processes'][pid]:
                        del self.config_data['processes'][pid][old_key]

            # 保存元数据
            self.config_data['processes'][pid]['groups_meta'] = new_groups_meta

            # 写入文件
            self._write_to_file()

            # 通知后台重载
            if self.controller:
                self.controller.reload_config()

            QMessageBox.information(self, "成功", "动作已保存，并已通知后台生效")
        except ValueError as e:
            QMessageBox.warning(self, "错误", f"数据格式错误: {e}")
        except Exception as e:
            QMessageBox.critical(self, "系统错误", f"保存失败: {str(e)}")

    def _write_to_file(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config_data, f, indent=4, ensure_ascii=False)

    def refresh_realtime_display(self):
        """定时器槽函数：从 Controller 获取最新数据并刷新界面"""
        if not self.controller:
            return

        # 直接读取 Controller 中的变量
        joint_values = self.controller.last_joint_status
        axis_values = self.controller.last_axis_status

        # 更新 Label 显示
        # 电机关节
        self.lbl_j1.setText(f"{joint_values[0]:.2f}")
        self.lbl_j2.setText(f"{joint_values[1]:.2f}")
        self.lbl_j3.setText(f"{joint_values[2]:.2f}")
        self.lbl_j4.setText(f"{joint_values[3]:.2f}")

        # 空间坐标
        self.lbl_x.setText(f"{axis_values[0]:.2f}")
        self.lbl_y.setText(f"{axis_values[1]:.2f}")
        self.lbl_z.setText(f"{axis_values[2]:.2f}")
        self.lbl_r.setText(f"{axis_values[3]:.2f}")

    def closeEvent(self, event):
        # if hasattr(self, 'monitor_thread'):
        #     self.monitor_thread.stop()
        self.ui_timer.stop()
        event.accept()

    # === 产品配置相关槽函数 ===
    def switch_product_model(self):
        """核心功能：切换产品型号"""
        selected_model = self.combo_product.currentText()
        current_model = self.lbl_current_product.text()

        # 1. 检查是否真的变化了
        if selected_model == current_model:
            QMessageBox.information(self, "提示", "当前已经是该型号，无需切换。")
            return

        # 2. 弹出警告对话框
        reply = QMessageBox.warning(
            self,
            "切换确认",
            f"确定要将生产型号切换为 [{selected_model}] 吗？\n\n注意：切换后程序需要重启以加载新产品的视觉和点位参数！",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 3. 更新内存数据
            if 'product_config' not in self.config_data:
                self.config_data['product_config'] = {}

            self.config_data['product_config']['current_model'] = selected_model

            # 确保列表也同步保存（防止用户添加了没保存直接点切换）
            model_list = [self.combo_product.itemText(i) for i in range(self.combo_product.count())]
            self.config_data['product_config']['model_list'] = model_list

            # 4. 写入文件
            self._write_to_file()

            # 5. 更新界面显示
            self.lbl_current_product.setText(selected_model)

            # 6. 提示重启
            QMessageBox.information(self, "切换成功", "产品型号已更新。\n\n请务必重启程序以确保所有参数生效！")

            # 可选：通知后台 (虽然提示了重启，但如果是热加载架构，也可以通知)
            if self.controller:
                self.controller.reload_config()

    def save_product_list_only(self):
        """只保存列表的增删，不改变当前型号"""
        model_list = [self.combo_product.itemText(i) for i in range(self.combo_product.count())]

        if 'product_config' not in self.config_data:
            self.config_data['product_config'] = {}

        self.config_data['product_config']['model_list'] = model_list
        self._write_to_file()
        QMessageBox.information(self, "成功", "型号列表已保存")

    def add_product_model(self):
        """添加新产品型号"""
        text, ok = QInputDialog.getText(self, "添加型号", "请输入新的产品型号 (例如 PN88888):")
        if ok and text:
            text = text.strip()
            if not text: return

            if self.combo_product.findText(text) >= 0:
                QMessageBox.warning(self, "提示", "该型号已存在！")
                return

            self.combo_product.addItem(text)
            self.combo_product.setCurrentIndex(self.combo_product.count() - 1)

    def delete_product_model(self):
        """删除当前选中的型号"""
        curr_idx = self.combo_product.currentIndex()
        if curr_idx < 0: return

        txt = self.combo_product.currentText()

        # 保护：不能删除当前正在生产的型号
        if txt == self.lbl_current_product.text():
            QMessageBox.warning(self, "禁止删除", f"[{txt}] 正在生产中，无法删除！\n请先切换到其他型号。")
            return

        reply = QMessageBox.question(self, "确认删除", f"确定要删除型号 [{txt}] 吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.combo_product.removeItem(curr_idx)

    def save_product_config(self):
        """保存产品配置到 JSON"""
        # 1. 获取列表中的所有型号
        model_list = []
        for i in range(self.combo_product.count()):
            model_list.append(self.combo_product.itemText(i))

        # 2. 获取当前选中的型号
        current_model = self.combo_product.currentText()

        # 3. 更新内存数据
        if 'product_config' not in self.config_data:
            self.config_data['product_config'] = {}

        self.config_data['product_config']['model_list'] = model_list
        self.config_data['product_config']['current_model'] = current_model

        # 4. 写入文件
        self._write_to_file()

        # 5. 通知后台
        if self.controller:
            self.controller.reload_config()  # 假设 Controller 有处理这个新字段的逻辑

        QMessageBox.information(self, "成功", f"产品配置已保存\n当前型号: {current_model}")

    def get_current_tool_model(self):
        """获取当前激活的夹具完整配置 dict"""
        tools = self.data.get("tools", {})
        curr_name = tools.get("current_model")
        models = tools.get("models", [])

        for m in models:
            if m["name"] == curr_name:
                return m
        return {}  # 或者返回默认

    def on_tool_selected(self, row):
        """点击列表，右侧显示详情"""
        if row < 0:
            # 清空输入框
            self.tool_name_edit.clear()
            self.tool_desc_edit.clear()
            self.cam_off_x.clear();
            self.cam_off_y.clear();
            self.cam_rot.clear()
            self.grip_off_x.clear();
            self.grip_off_y.clear();
            self.grip_z_diff.clear()
            return

        tool_name = self.list_tools.item(row).text()

        # 从配置中查找对应的数据
        tools_cfg = self.config_data.get('tools', {})
        models = tools_cfg.get('models', [])

        target_data = next((m for m in models if m['name'] == tool_name), None)

        if target_data:
            self.tool_name_edit.setText(target_data.get('name', ''))

            grip_data = target_data.get('main_gripper', {})
            self.tool_desc_edit.setText(grip_data.get('desc', ''))
            self.grip_off_x.setText(str(grip_data.get('offset_x', 0)))
            self.grip_off_y.setText(str(grip_data.get('offset_y', 0)))
            self.grip_z_diff.setText(str(grip_data.get('z_diff', 0)))

            cam_data = target_data.get('camera', {})
            self.cam_off_x.setText(str(cam_data.get('offset_x', 0)))
            self.cam_off_y.setText(str(cam_data.get('offset_y', 0)))
            self.cam_rot.setText(str(cam_data.get('rotation', 0)))

    # === 工装夹具相关槽函数 ===

    def switch_tool_model(self):
        """切换当前使用的夹具"""
        selected_tool = self.combo_tools.currentText()
        current_tool = self.lbl_current_tool.text()

        if selected_tool == current_tool:
            return

        reply = QMessageBox.warning(
            self, "切换确认",
            f"确定要切换工装夹具为 [{selected_tool}] 吗？\n\n请确认物理硬件已更换完毕！",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if 'tools' not in self.config_data:
                self.config_data['tools'] = {}

            self.config_data['tools']['current_model'] = selected_tool
            self._write_to_file()

            self.lbl_current_tool.setText(selected_tool)

            if self.controller:
                self.controller.reload_config()

            QMessageBox.information(self, "成功", "工装配置已更新")

    def add_tool_model(self):
        """添加新夹具"""
        text, ok = QInputDialog.getText(self, "新建夹具", "请输入夹具名称 (例如: 3号夹具):")
        if ok and text:
            text = text.strip()
            if not text: return

            # 查重
            models = self.config_data.get('tools', {}).get('models', [])
            if any(m['name'] == text for m in models):
                QMessageBox.warning(self, "错误", "该夹具名称已存在")
                return

            # 创建默认结构
            new_model = {
                "name": text,
                "camera": {"offset_x": 0.0, "offset_y": 0.0, "rotation": 0},
                "main_gripper": {"desc": "默认夹爪", "offset_x": 0.0, "offset_y": 0.0, "z_diff": 0.0}
            }

            # 保存
            if 'tools' not in self.config_data:
                self.config_data['tools'] = {'models': [], 'current_model': ''}

            self.config_data['tools']['models'].append(new_model)
            self._write_to_file()

            # 刷新列表并选中
            self.load_config_from_file()  # 重新加载最简单，确保 Combo 和 List 同步

            # 选中新加的项
            items = self.list_tools.findItems(text, Qt.MatchExactly)
            if items:
                self.list_tools.setCurrentItem(items[0])

    def delete_tool_model(self):
        """删除夹具"""
        row = self.list_tools.currentRow()
        if row < 0: return

        tool_name = self.list_tools.item(row).text()
        current_tool = self.lbl_current_tool.text()

        # 【关键】禁止删除当前正在使用的
        if tool_name == current_tool:
            QMessageBox.warning(self, "禁止删除", f"[{tool_name}] 正在使用中，无法删除！\n请先切换到其他夹具。")
            return

        reply = QMessageBox.question(self, "确认删除", f"确定要永久删除 [{tool_name}] 的配置吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            models = self.config_data.get('tools', {}).get('models', [])
            # 过滤掉要删除的
            new_models = [m for m in models if m['name'] != tool_name]
            self.config_data['tools']['models'] = new_models

            self._write_to_file()
            self.load_config_from_file()  # 刷新界面

    def save_current_tool_params(self):
        """保存右侧编辑的参数"""
        reply = QMessageBox.question(self, "确认保存修改",
                                     f"确定要修改参数吗?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:

            row = self.list_tools.currentRow()
            if row < 0: return

            tool_name = self.list_tools.item(row).text()

            try:
                # 找到内存中的引用
                models = self.config_data.get('tools', {}).get('models', [])
                target_data = next((m for m in models if m['name'] == tool_name), None)

                if target_data:
                    # 更新数据
                    target_data['main_gripper']['desc'] = self.tool_desc_edit.text()
                    target_data['main_gripper']['offset_x'] = float(self.grip_off_x.text())
                    target_data['main_gripper']['offset_y'] = float(self.grip_off_y.text())
                    target_data['main_gripper']['z_diff'] = float(self.grip_z_diff.text())

                    target_data['camera']['offset_x'] = float(self.cam_off_x.text())
                    target_data['camera']['offset_y'] = float(self.cam_off_y.text())
                    target_data['camera']['rotation'] = float(self.cam_rot.text())

                    self._write_to_file()

                    # 如果修改的是当前正在使用的夹具，需要通知后台重载
                    if tool_name == self.lbl_current_tool.text():
                        if self.controller:
                            self.controller.reload_config()

                    QMessageBox.information(self, "成功", "参数已保存")

            except ValueError:
                QMessageBox.warning(self, "错误", "请输入有效的数字参数")


# === 测试入口 ===
if __name__ == "__main__":
    # 模拟 PLC Client 用于测试界面
    class MockPLC:
        is_connected = True

        def read_holding_registers(self, addr, count):
            # 模拟返回随机数据
            import random
            return [random.randint(0, 100) for _ in range(count)]

        def registers_to_float(self, regs):
            return regs[0] + regs[1] / 100.0


    app = QApplication(sys.argv)
    window = ConfigEditorUI(plc_client=MockPLC())
    window.show()
    sys.exit(app.exec_())
