# -*- coding: utf-8 -*-
import sys
import json
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QFormLayout, QLabel, QLineEdit, QPushButton,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                             QSplitter, QListWidget, QMessageBox, QComboBox, QInputDialog, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from src.utils.config_manager import CONFIG_FILE

ADDR_FEEDBACK_START = 0x400BE
FEEDBACK_LEN = 8  # 4个float = 8个寄存器

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
        tabs.addTab(self.tab_process, "动作流程(Process)")

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

        self.list_processes = QListWidget()
        self.list_processes.currentRowChanged.connect(self.on_process_selected)
        left_layout.addWidget(self.list_processes)

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

        # 点位表格
        self.table_points = QTableWidget()
        self.table_points.setColumnCount(7)
        self.table_points.setHorizontalHeaderLabels(["点名称", "X", "Y", "Z", "R (te)", "Photo", "姿态"])
        header = self.table_points.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        right_layout.addWidget(self.table_points)

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

    # === 逻辑处理部分 ===
    def _set_row_elbow_combo(self, row, current_val="elbow_up"):
        """辅助函数：给指定行设置姿态下拉框"""
        combo_config = QComboBox()
        combo_config.addItems(["elbow_up", "elbow_down"])

        idx = combo_config.findText(current_val)
        if idx >= 0:
            combo_config.setCurrentIndex(idx)
        else:
            combo_config.setCurrentIndex(0)  # 默认

        self.table_points.setCellWidget(row, 6, combo_config)

    def swap_table_rows(self, row1, row2):
        """核心逻辑：交换两行的数据"""
        # 1. 交换普通单元格 (0-5列)
        for col in range(6):
            item1 = self.table_points.takeItem(row1, col)
            item2 = self.table_points.takeItem(row2, col)
            self.table_points.setItem(row2, col, item1)
            self.table_points.setItem(row1, col, item2)

        # 2. 交换下拉框 (第6列/index 6)
        # Qt 的 setCellWidget 会销毁旧的 widget，所以必须读取值后重新创建
        combo1 = self.table_points.cellWidget(row1, 6)
        combo2 = self.table_points.cellWidget(row2, 6)

        val1 = combo1.currentText() if combo1 else "elbow_up"
        val2 = combo2.currentText() if combo2 else "elbow_up"

        self._set_row_elbow_combo(row2, val1)
        self._set_row_elbow_combo(row1, val2)

    def move_point_up(self):
        """上移当前行"""
        row = self.table_points.currentRow()
        # 如果没选中，或者已经是第一行，无法上移
        if row <= 0:
            return

        # 交换当前行与上一行
        self.swap_table_rows(row, row - 1)

        # 保持选中状态跟随移动
        self.table_points.setCurrentCell(row - 1, 0)

    def move_point_down(self):
        """下移当前行"""
        row = self.table_points.currentRow()
        count = self.table_points.rowCount()

        # 如果没选中，或者是最后一行，无法下移
        if row < 0 or row >= count - 1:
            return

        # 交换当前行与下一行
        self.swap_table_rows(row, row + 1)

        # 保持选中状态跟随移动
        self.table_points.setCurrentCell(row + 1, 0)

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

            # 填充动作列表
            self.list_processes.clear()
            processes = self.config_data.get('processes', {})
            for pid in processes.keys():
                self.list_processes.addItem(pid)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载配置文件失败: {e}")

    def on_process_selected(self, row):
        """刷新右侧显示"""
        if row < 0:
            # 如果没有选中项，清空右侧
            self.lbl_proc_name.clear()
            self.lbl_proc_type.clear()
            self.lbl_proc_d_addr.clear()
            self.table_points.setRowCount(0)
            return

        pid = self.list_processes.item(row).text()
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

        points = process_data.get('points', [])
        self.table_points.setRowCount(len(points))

        # 临时关闭排序功能，防止填充时乱跳
        self.table_points.setSortingEnabled(False)

        for i, pt in enumerate(points):
            print("#######################")
            print(pt)
            self.table_points.setItem(i, 0, QTableWidgetItem(str(pt.get('name', ''))))
            coords = pt.get('coords', [0, 0, 0, 0])
            for j in range(4):
                # 格式化一下，保留2位小数显示
                val = f"{coords[j]:.2f}"
                self.table_points.setItem(i, j + 1, QTableWidgetItem(val))
            self.table_points.setItem(i, 5, QTableWidgetItem(str(pt.get('photo', 0))))

            #  === 【新增】设置姿态下拉框 ===
            current_config = pt.get('config', 'elbow_up')

            # combo_config = QComboBox()
            # combo_config.addItems(["elbow_up", "elbow_down"])
            #
            # # 获取当前点的配置，默认为 up
            # # 选中对应的值
            # index = combo_config.findText(current_config)
            # if index >= 0:
            #     combo_config.setCurrentIndex(index)
            # else:
            #     combo_config.setCurrentIndex(0)  # 默认 up
            #
            # # 将下拉框放入单元格 (第6列，即第7个)
            # self.table_points.setCellWidget(i, 6, combo_config)

            self._set_row_elbow_combo(i, current_val=current_config)

        self.table_points.setSortingEnabled(False)  # 表格通常不需要排序，容易乱序


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

            # 2. 初始化数据结构
            new_process = {
                "name": "新建动作流程",
                "type": "standard",
                "points": []
            }

            # 3. 更新内存数据
            if 'processes' not in self.config_data:
                self.config_data['processes'] = {}
            self.config_data['processes'][text] = new_process

            # 4. 更新 UI 列表
            self.list_processes.addItem(text)
            # 选中新添加的项
            self.list_processes.setCurrentRow(self.list_processes.count() - 1)

            # 5. 自动保存（可选，或者让用户点保存按钮）
            self._write_to_file()

    def delete_process_item(self):
        """删除当前选中的动作"""
        row = self.list_processes.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先选择要删除的动作")
            return

        pid = self.list_processes.item(row).text()

        # 二次确认
        reply = QMessageBox.question(self, "确认删除",
                                     f"确定要永久删除动作 [{pid}] 及其所有点位吗？",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 1. 从内存移除
            if pid in self.config_data.get('processes', {}):
                del self.config_data['processes'][pid]

            # 2. 从 UI 移除
            self.list_processes.takeItem(row)

            # 3. 自动保存
            self._write_to_file()

            # 4. 通知后台刷新
            if self.controller:
                self.controller.reload_config()

    def teach_current_position(self):
        """示教功能：读取当前电机坐标并填入表格"""
        if not self.controller:
            return

        # 从 Controller 获取实时缓存 [j1, j2, j3, j4]
        # 注意：这里拿到的是关节角度/高度，需要根据你的业务需求决定填什么
        # 如果你的 points 存的是 [x, y, z, r] (笛卡尔)，你需要在这里调用正运动学转一下
        # 假设 Controller 有 get_realtime_point 返回的是 {coords: [x,y,z,r]}

        try:
            real_point = self.controller.get_realtime_point()
            if not real_point:
                QMessageBox.warning(self, "错误", "无法获取当前机械臂坐标")
                return

            x, y, z, r = real_point['coords']
            # 获取实时的姿态 (elbow_up/down)
            current_elbow = real_point.get('config', 'elbow_up')

            # 添加新行
            # row = self.table_points.rowCount()
            # self.table_points.insertRow(row)

            current_row = self.table_points.currentRow()
            if current_row >= 0:
                insert_idx = current_row + 1
            else:
                insert_idx = self.table_points.rowCount()

            self.table_points.insertRow(insert_idx)

            self.table_points.setItem(insert_idx, 0, QTableWidgetItem(f"Teach_P{insert_idx + 1}"))
            self.table_points.setItem(insert_idx, 1, QTableWidgetItem(f"{x:.2f}"))
            self.table_points.setItem(insert_idx, 2, QTableWidgetItem(f"{y:.2f}"))
            self.table_points.setItem(insert_idx, 3, QTableWidgetItem(f"{z:.2f}"))
            self.table_points.setItem(insert_idx, 4, QTableWidgetItem(f"{r:.2f}"))
            self.table_points.setItem(insert_idx, 5, QTableWidgetItem("0"))

            # === 【新增】设置示教的姿态 ===
            # combo_config = QComboBox()
            # combo_config.addItems(["elbow_up", "elbow_down"])
            #
            # # 自动选中当前机械臂的姿态
            # idx = combo_config.findText(current_elbow)
            # if idx >= 0:
            #     combo_config.setCurrentIndex(idx)
            #
            # self.table_points.setCellWidget(insert_idx, 6, combo_config)

            self._set_row_elbow_combo(insert_idx, current_val=current_elbow)
            # 自动选中新行
            self.table_points.setCurrentCell(insert_idx, 0)

        except Exception as e:
            QMessageBox.warning(self, "异常", f"示教失败: {e}")

    def add_point_row(self):
        current_row = self.table_points.currentRow()
        if current_row >= 0:
            # 如果有选中行，插在选中行的下一行
            insert_idx = current_row + 1
        else:
            # 如果没选中，插在表格末尾
            insert_idx = self.table_points.rowCount()

        # row = self.table_points.rowCount()
        self.table_points.insertRow(insert_idx)
        # 设置默认值
        self.table_points.setItem(insert_idx, 0, QTableWidgetItem(f"P{insert_idx + 1}"))
        for j in range(1, 6):
            self.table_points.setItem(insert_idx, j, QTableWidgetItem("0"))

        # combo_config = QComboBox()
        # combo_config.addItems(["elbow_up", "elbow_down"])
        # self.table_points.setCellWidget(insert_idx, 6, combo_config)

        self._set_row_elbow_combo(insert_idx)

        # 自动选中新添加的行
        self.table_points.setCurrentCell(insert_idx, 0)

    def delete_point_row(self):
        current_row = self.table_points.currentRow()
        if current_row >= 0:
            self.table_points.removeRow(current_row)

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
        current_item = self.list_processes.currentItem()
        if not current_item:
            return

        pid = current_item.text()
        new_points = []

        try:
            rows = self.table_points.rowCount()
            for i in range(rows):
                name = self.table_points.item(i, 0).text()
                x = float(self.table_points.item(i, 1).text())
                y = float(self.table_points.item(i, 2).text())
                z = float(self.table_points.item(i, 3).text())
                r = float(self.table_points.item(i, 4).text())
                photo = int(self.table_points.item(i, 5).text())

                combo = self.table_points.cellWidget(i, 6)
                if combo:
                    config_val = combo.currentText()
                else:
                    config_val = "elbow_up"  # 默认值防止报错

                new_points.append({
                    "name": name,
                    "coords": [x, y, z, r],
                    "photo": photo,
                    "config": config_val
                })

            # 更新内存数据
            self.config_data['processes'][pid]['name'] = self.lbl_proc_name.text()
            self.config_data['processes'][pid]['type'] = self.lbl_proc_type.text()

            try:
                d_addr_val = int(self.lbl_proc_d_addr.text())
                self.config_data['processes'][pid]['d_addr'] = d_addr_val
            except ValueError:
                pass  # 如果是 Error 字符串或其他非数字，则不保存

            self.config_data['processes'][pid]['points'] = new_points

            self._write_to_file()

            # 通知后台重载
            if self.controller:
                self.controller.reload_config()

            QMessageBox.information(self, "成功", "动作已保存，并已通知后台生效")
        except ValueError as e:
            QMessageBox.warning(self, "错误", f"数据格式错误: {e}")

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