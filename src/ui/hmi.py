# -*- coding: utf-8 -*-
"""
SCARA 智能视觉上下料工作站 — HMI 主仪表盘
=======================================

设计目标:
- 10.1" 触控屏 (1280×800)
- 工人视角：当前动作、相机画面、检测结果一屏可见
- 客户宣讲：暗色专业风格，大画面视觉冲击

数据交互说明 (只做计划、不执行):
  参见 _update_ui_data 和 _on_camera_frame 方法的注释。

启动方式:
    uv run python -m src.ui.hmi              # 独立测试
    uv run python main.py                     # 正式运行 (由 main.py 创建)
"""
import sys
import os
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFrame, QDialog,
    QLineEdit, QFormLayout, QDialogButtonBox, QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer, QDateTime, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QFontDatabase, QIcon, QImage


# ======================================================================
#  辅助类
# ======================================================================

class ClickableLabel(QLabel):
    """可点击的图片标签，用于相机画面 → 全屏查看

    特性:
      - 保持原始 pixmap 引用，全屏时显示原始尺寸
      - 显示时按比例缩放填满标签区域，不撑大布局
      - sizeHint 返回 (1,1) 防止影响父布局尺寸
    """
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1, 1)
        self._origin_pixmap = None

    def sizeHint(self):
        from PyQt5.QtCore import QSize
        return QSize(1, 1)

    def minimumSizeHint(self):
        from PyQt5.QtCore import QSize
        return QSize(1, 1)

    def set_image(self, pixmap):
        """设置图片，保存原始引用并按比例缩放显示"""
        self._origin_pixmap = pixmap
        self._update_display()

    def origin_pixmap(self):
        """返回原始 (未缩放) pixmap，用于全屏查看"""
        return self._origin_pixmap

    def _update_display(self):
        if self._origin_pixmap is None or self._origin_pixmap.isNull():
            super().setPixmap(QPixmap())
            return
        scaled = self._origin_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class FullscreenOverlay(QDialog):
    """全屏图片查看弹窗 (黑色背景，点击/Esc/30s 自动关闭)"""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0.9);")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 图片居中显示 (保持比例)
        self.image_label = QLabel()
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.image_label, 1)

        # 底部提示文字
        hint = QLabel("点击画面、按 ESC 或等待 30秒 退出全屏")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("""
            color: rgba(255, 255, 255, 0.6);
            font-size: 14px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 20px;
            padding: 8px 16px;
            margin-bottom: 30px;
        """)
        layout.addWidget(hint, 0, Qt.AlignCenter)

        self.showFullScreen()

        QTimer.singleShot(30000, self.close)

    def mousePressEvent(self, event):
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


class StatusDot(QFrame):
    """状态指示灯圆点 (12×12, 绿/红)"""

    def __init__(self, size: int = 12, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._r = size // 2
        self.set_green()

    def set_green(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #10b981;
                border-radius: {self._r}px;
            }}
        """)

    def set_red(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #ef4444;
                border-radius: {self._r}px;
            }}
        """)

    def set_by_bool(self, ok: bool):
        if ok:
            self.set_green()
        else:
            self.set_red()


class PasswordDialog(QDialog):
    """用户名/密码验证弹窗 (验证通过后打开 ConfigEditorUI)"""

    # 简单硬编码凭证 (可改为配置或环境变量)
    _USERNAME = "admin"
    _PASSWORD = "123456"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("身份验证 — 系统配置")
        self.setFixedSize(360, 200)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("用户名")
        form.addRow("用户名:", self.username_edit)

        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("密码")
        self.password_edit.setEchoMode(QLineEdit.Password)
        form.addRow("密码:", self.password_edit)

        layout.addLayout(form)
        layout.addStretch()

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._validate)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _validate(self):
        if (self.username_edit.text().strip() == self._USERNAME
                and self.password_edit.text() == self._PASSWORD):
            self.accept()
        else:
            QMessageBox.warning(self, "验证失败", "用户名或密码错误，请重试。")
            self.password_edit.clear()
            self.password_edit.setFocus()


# ======================================================================
#  主仪表盘
# ======================================================================

class MainHMI(QMainWindow):
    """SCARA 工作站 HMI 主仪表盘 (10.1" 触控屏优化)"""

    # --- 占位默认值 ---
    _PLACEHOLDER_COLOR = "#1a1a2e"
    _PLACEHOLDER_RGB_TEXT = "RGB 原图"
    _PLACEHOLDER_DEPTH_TEXT = "识别 / 热力图"

    def __init__(self, controller=None, parent=None):
        super().__init__(parent)
        self.controller = controller

        self.setWindowTitle("SCARA 智能视觉上下料工作站")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 600)

        # 内部引用
        self._config_window = None
        self._fullscreen_overlay = None

        # 占位图缓存
        self._placeholder_rgb = None
        self._placeholder_depth = None

        self._init_ui()
        self._init_timers()
        self._apply_stylesheet()
        self._connect_signals()

    # ================================================================
    #  UI 构建
    # ================================================================

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Header
        header = self._create_header()
        main_layout.addWidget(header)

        # Body
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(16)

        left_panel = self._create_left_panel()    # 60%
        right_panel = self._create_right_panel()  # 40%

        body_layout.addWidget(left_panel, 6)
        body_layout.addWidget(right_panel, 4)

        main_layout.addWidget(body, 1)

    # ---------- Header ----------

    def _create_header(self):
        header = QFrame()
        header.setObjectName("header")
        header.setStyleSheet("""
            QFrame#header {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
        """)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # Logo
        logo = QLabel("S")
        logo.setFixedSize(32, 32)
        logo.setStyleSheet("""
            background-color: #059669;
            color: white;
            font-weight: bold;
            font-size: 16px;
            border-radius: 4px;
        """)
        logo.setAlignment(Qt.AlignCenter)

        # Title
        title = QLabel("SCARA 智能视觉上下料工作站")
        title.setStyleSheet("""
            font-weight: bold;
            font-size: 16px;
            color: #1f2937;
        """)

        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addStretch()

        # 当前型号
        self.model_label = QLabel("当前型号：PN12345")
        self.model_label.setStyleSheet("font-size: 13px; color: #4b5563;")

        # 时钟
        self.clock_label = QLabel("--:--:--")
        self.clock_label.setStyleSheet("""
            font-size: 13px;
            color: #dc2626;
            font-family: monospace;
            font-weight: bold;
        """)

        # 齿轮按钮
        gear_btn = QPushButton()
        gear_btn.setFixedSize(36, 36)
        gear_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 4px;
                font-size: 20px;
            }
            QPushButton:hover { background: #f3f4f6; }
        """)
        gear_btn.setText("\u2699")
        gear_btn.setToolTip("系统配置 (需身份验证)")
        gear_btn.clicked.connect(self._open_config)

        layout.addWidget(self.model_label)
        layout.addSpacing(16)
        layout.addWidget(self.clock_label)
        layout.addSpacing(8)
        layout.addWidget(gear_btn)

        return header

    # ---------- 左侧：视觉图像区域 ----------

    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # RGB 原图
        self.rgb_card, self.rgb_image = self._create_image_card(
            title="RGB 原图",
            bg_color="#1e1e2e",
            text="RGB 图像"
        )
        layout.addWidget(self.rgb_card, 1)

        # 识别/热力图
        self.depth_card, self.depth_image = self._create_image_card(
            title="识别 / 热力图",
            bg_color="#1a365d",
            text="深度 / 检测结果"
        )
        layout.addWidget(self.depth_card, 1)

        return panel

    def _create_image_card(self, title, bg_color, text):
        """创建视觉图像卡片 (图片 + 左上角标题徽标)"""
        card = QFrame()
        card.setObjectName("image_card")
        card.setStyleSheet(f"""
            QFrame#image_card {{
                background-color: {bg_color};
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }}
        """)

        # 使用网格布局实现图片铺满 + 徽标叠加
        grid = QGridLayout(card)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        # 图片标签 (填满，不撑大布局)
        image = ClickableLabel()
        placeholder = self._make_placeholder(
            width=640, height=360,
            bg=bg_color,
            text=text,
        )
        image.set_image(placeholder)
        image.clicked.connect(lambda: self._show_fullscreen(image.origin_pixmap()))
        grid.addWidget(image, 0, 0)
        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)

        # 徽标 (叠加在左上角)
        badge = QLabel(title)
        badge.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 4px;
        """)
        badge.adjustSize()
        grid.addWidget(badge, 0, 0, Qt.AlignLeft | Qt.AlignTop)

        return card, image

    def _make_placeholder(self, width, height, bg, text):
        """创建占位 QPixmap (用于无相机时界面测试)"""
        pix = QPixmap(width, height)
        pix.fill(QColor(bg))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(160, 160, 180))
        font = QFont("Microsoft YaHei", 22)
        painter.setFont(font)
        painter.drawText(pix.rect(), Qt.AlignCenter, text)
        painter.end()
        return pix

    # ---------- 右侧：监控面板 ----------

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # KPI 看板 (固定高度)
        kpi_card = self._create_kpi_card()
        layout.addWidget(kpi_card)

        # 5 行卡片容器
        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(6)

        cards_layout.addWidget(self._create_process_card(), 1)
        cards_layout.addWidget(self._create_action_card(), 1)
        cards_layout.addWidget(self._create_coords_card(), 1)
        cards_layout.addWidget(self._create_joints_card(), 1)
        cards_layout.addWidget(self._create_hw_card(), 1)

        layout.addWidget(cards_container, 1)

        return panel

    # --- Card 0: KPI 看板 ---

    def _create_kpi_card(self):
        card = QFrame()
        card.setObjectName("kpi_card")
        card.setStyleSheet("""
            QFrame#kpi_card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
        """)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(0)

        # 节拍
        cycle_widget = QWidget()
        cycle_layout = QVBoxLayout(cycle_widget)
        cycle_layout.setContentsMargins(0, 0, 0, 0)
        cycle_layout.setSpacing(2)
        cycle_layout.setAlignment(Qt.AlignCenter)

        self.kpi_cycle_value = QLabel("8.5s")
        self.kpi_cycle_value.setStyleSheet("""
            font-size: 30px;
            font-weight: bold;
            color: #10b981;
        """)
        self.kpi_cycle_value.setAlignment(Qt.AlignCenter)

        self.kpi_cycle_label = QLabel("当前节拍")
        self.kpi_cycle_label.setStyleSheet("font-size: 11px; color: #6b7280;")
        self.kpi_cycle_label.setAlignment(Qt.AlignCenter)

        cycle_layout.addWidget(self.kpi_cycle_value)
        cycle_layout.addWidget(self.kpi_cycle_label)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("border: none; border-left: 1px solid #e5e7eb;")

        # 产量
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(2)
        output_layout.setAlignment(Qt.AlignCenter)

        self.kpi_output_value = QLabel("1,250")
        self.kpi_output_value.setStyleSheet("""
            font-size: 30px;
            font-weight: bold;
            color: #10b981;
        """)
        self.kpi_output_value.setAlignment(Qt.AlignCenter)

        self.kpi_output_label = QLabel("今日产量")
        self.kpi_output_label.setStyleSheet("font-size: 11px; color: #6b7280;")
        self.kpi_output_label.setAlignment(Qt.AlignCenter)

        output_layout.addWidget(self.kpi_output_value)
        output_layout.addWidget(self.kpi_output_label)

        layout.addWidget(cycle_widget, 1)
        layout.addWidget(separator)
        layout.addWidget(output_widget, 1)

        return card

    # --- Card 1: 工艺进度 ---

    def _create_process_card(self):
        card = QFrame()
        card.setObjectName("process_card")
        card.setStyleSheet("""
            QFrame#process_card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # 标题行
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("\U0001f4cd 工艺进度")
        title.setStyleSheet("font-size: 11px; color: #2563eb; font-weight: bold;")

        self.process_current_label = QLabel("当前：  ")
        self.process_current_label.setStyleSheet("""
            font-size: 12px;
            color: #059669;
            font-weight: bold;
        """)

        title_layout.addWidget(title)
        title_layout.addStretch()
        title_layout.addWidget(self.process_current_label)

        layout.addLayout(title_layout)

        # 5 个步骤
        steps_layout = QHBoxLayout()
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(4)

        step_labels = ["工装除屑", "上料", "下料", "检测", "码垛"]
        self._step_widgets = []

        for i, label in enumerate(step_labels):
            step = self._make_step(label)
            self._step_widgets.append(step)
            steps_layout.addWidget(step, 1)

        layout.addLayout(steps_layout)

        return card

    def _make_step(self, text):
        """创建单个工艺步骤小部件 (含圆点 + 文字)"""
        widget = QFrame()
        widget.setObjectName("step_inactive")
        widget.setStyleSheet("""
            QFrame#step_inactive {
                background: #f3f4f6;
                border: 2px solid transparent;
                border-radius: 6px;
            }
        """)

        vlay = QVBoxLayout(widget)
        vlay.setContentsMargins(2, 4, 2, 4)
        vlay.setSpacing(2)
        vlay.setAlignment(Qt.AlignCenter)

        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet("background: #d1d5db; border-radius: 5px;")

        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 10px; color: #9ca3af;")
        lbl.setAlignment(Qt.AlignCenter)

        vlay.addWidget(dot, 0, Qt.AlignCenter)
        vlay.addWidget(lbl, 0, Qt.AlignCenter)

        return widget

    def set_process_step(self, index: int):
        """高亮第 index 个步骤 (0-based)"""
        for i, w in enumerate(self._step_widgets):
            if i == index:
                w.setObjectName("step_active")
                w.setStyleSheet("""
                    QFrame#step_active {
                        background: #10b981;
                        border: 2px solid #047857;
                        border-radius: 6px;
                    }
                """)
                # 更新内部子控件
                dot = w.findChild(QLabel)
                if dot:
                    dot.setStyleSheet("background: white; border-radius: 5px;")
                lbl = w.findChildren(QLabel)
                if len(lbl) >= 2:
                    lbl[1].setStyleSheet("font-size: 10px; color: white; font-weight: bold;")
                w.style().unpolish(w)
                w.style().polish(w)
            else:
                w.setObjectName("step_inactive")
                w.setStyleSheet("""
                    QFrame#step_inactive {
                        background: #f3f4f6;
                        border: 2px solid transparent;
                        border-radius: 6px;
                    }
                """)
                dot = w.findChild(QLabel)
                if dot:
                    dot.setStyleSheet("background: #d1d5db; border-radius: 5px;")
                lbl = w.findChildren(QLabel)
                if len(lbl) >= 2:
                    lbl[1].setStyleSheet("font-size: 10px; color: #9ca3af;")
                w.style().unpolish(w)
                w.style().polish(w)

    # --- Card 2: 当前动作 ---

    def _create_action_card(self):
        card = QFrame()
        card.setObjectName("action_card")
        card.setStyleSheet("""
            QFrame#action_card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        title = QLabel("\U0001f916 当前动作")
        title.setStyleSheet("font-size: 11px; color: #2563eb; font-weight: bold;")
        layout.addWidget(title)

        self.action_text = QLabel(">>> 正在执行：端头视觉精拍...")
        self.action_text.setStyleSheet("""
            font-size: 12px;
            color: #1f2937;
            font-family: monospace;
            background: #f9fafb;
            padding: 6px 10px;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        """)
        self.action_text.setWordWrap(True)
        layout.addWidget(self.action_text)

        return card

    # --- Card 3: 末端坐标 ---

    def _create_coords_card(self):
        card = QFrame()
        card.setObjectName("coords_card")
        card.setStyleSheet("""
            QFrame#coords_card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        title = QLabel("\U0001f4d0 末端坐标")
        title.setStyleSheet("font-size: 11px; color: #2563eb; font-weight: bold;")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(4)

        self.coord_labels = {}
        coord_names = [("X", 0, 0), ("Y", 0, 1), ("Z", 1, 0), ("R", 1, 1)]

        for name, row, col in coord_names:
            container = QFrame()
            container.setStyleSheet("background: #f9fafb; border-radius: 4px; padding: 4px;")
            clayout = QHBoxLayout(container)
            clayout.setContentsMargins(6, 3, 6, 3)
            clayout.setSpacing(4)

            nlabel = QLabel(f"{name}:")
            nlabel.setStyleSheet("font-size: 11px; color: #4b5563;")
            vlabel = QLabel("0.0")
            vlabel.setStyleSheet("font-size: 12px; font-weight: bold; color: #111827;")

            clayout.addWidget(nlabel)
            clayout.addWidget(vlabel, 1, Qt.AlignRight)

            self.coord_labels[name] = vlabel
            grid.addWidget(container, row, col)

        layout.addLayout(grid)

        return card

    # --- Card 4: 电机关节 ---

    def _create_joints_card(self):
        card = QFrame()
        card.setObjectName("joints_card")
        card.setStyleSheet("""
            QFrame#joints_card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        title = QLabel("\u26a1 电机关节")
        title.setStyleSheet("font-size: 11px; color: #2563eb; font-weight: bold;")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(4)

        self.joint_labels = {}
        joint_names = [("J1", 0, 0), ("J2", 0, 1), ("J3", 1, 0), ("J4", 1, 1)]

        for name, row, col in joint_names:
            container = QFrame()
            container.setStyleSheet("background: #f9fafb; border-radius: 4px; padding: 4px;")
            clayout = QHBoxLayout(container)
            clayout.setContentsMargins(6, 3, 6, 3)
            clayout.setSpacing(4)

            nlabel = QLabel(f"{name}:")
            nlabel.setStyleSheet("font-size: 11px; color: #4b5563;")
            vlabel = QLabel("0.0")
            vlabel.setStyleSheet("font-size: 12px; font-weight: bold; color: #111827;")

            clayout.addWidget(nlabel)
            clayout.addWidget(vlabel, 1, Qt.AlignRight)

            self.joint_labels[name] = vlabel
            grid.addWidget(container, row, col)

        layout.addLayout(grid)

        return card

    # --- Card 5: 硬件状态 ---

    def _create_hw_card(self):
        card = QFrame()
        card.setObjectName("hw_card")
        card.setStyleSheet("""
            QFrame#hw_card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
        """)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(0)

        title = QLabel("\u2699\ufe0f 硬件状态")
        title.setStyleSheet("font-size: 11px; color: #2563eb; font-weight: bold;")
        layout.addWidget(title)

        layout.addSpacing(24)

        # PLC
        plc_widget = QWidget()
        plc_layout = QHBoxLayout(plc_widget)
        plc_layout.setContentsMargins(0, 0, 0, 0)
        plc_layout.setSpacing(6)

        self.plc_dot = StatusDot(12)
        plc_text = QLabel("PLC 控制器")
        plc_text.setStyleSheet("font-size: 12px; color: #374151;")

        plc_layout.addWidget(self.plc_dot)
        plc_layout.addWidget(plc_text)

        # 相机
        cam_widget = QWidget()
        cam_layout = QHBoxLayout(cam_widget)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        cam_layout.setSpacing(6)

        self.cam_dot = StatusDot(12)
        cam_text = QLabel("工业相机")
        cam_text.setStyleSheet("font-size: 12px; color: #374151;")

        cam_layout.addWidget(self.cam_dot)
        cam_layout.addWidget(cam_text)

        layout.addWidget(plc_widget)
        layout.addSpacing(16)
        layout.addWidget(cam_widget)
        layout.addStretch()

        return card

    # ================================================================
    #  全局样式
    # ================================================================

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
        """)

    # ================================================================
    #  信号连接 (Controller → HMI)
    # ================================================================

    def _connect_signals(self):
        """连接 Controller 信号 → HMI 槽函数"""
        if self.controller is None:
            return
        # 相机画面信号 (跨线程安全，由 Controller QThread 发射)
        if hasattr(self.controller, 'sig_camera_frame'):
            self.controller.sig_camera_frame.connect(self._on_camera_frame)

        # 生产 KPI 信号 (今日产量 / 当前节拍, 每次放料成功 +1 时由 Controller 发射)
        if hasattr(self.controller, 'sig_kpi'):
            self.controller.sig_kpi.connect(self._on_kpi)
            # 启动时立即用持久化的历史数据刷新, 避免重启后要等下一次放料才显示产量
            count, cycle_time = self.controller.kpi_counter.get_snapshot()
            self._apply_kpi(count, cycle_time)

    def _on_camera_frame(self, color_img, result_img):
        """
        接收 Controller 传来的相机帧，更新左侧图像显示。

        调用链:
            Controller.take_photo_position()
              → vision_service.execute_detection_midian_depth()
                → execute_detection()  缓存 _last_color_img / _last_result_img
              → Controller.sig_camera_frame.emit()
              → MainHMI._on_camera_frame()  (主线程)

        数据类型:
            color_img  — numpy.ndarray (H, W, 3)  RGB uint8  原图
            result_img — numpy.ndarray (H, W, 3)  BGR uint8  检测结果叠加图
        """
        if color_img is not None:
            self._set_pixmap_from_ndarray(self.rgb_image, color_img, is_bgr=False)
        if result_img is not None:
            self._set_pixmap_from_ndarray(self.depth_image, result_img, is_bgr=True)

    def _on_kpi(self, count, cycle_time):
        """接收 Controller 的产量/节拍信号, 更新 KPI 卡片 (主线程执行)"""
        self._apply_kpi(count, cycle_time)

    def _apply_kpi(self, count, cycle_time):
        """刷新 KPI 卡片数值: 今日产量(千分位) 与 当前节拍"""
        self.kpi_output_value.setText(f"{count:,}")
        if cycle_time is not None and cycle_time > 0:
            self.kpi_cycle_value.setText(f"{cycle_time:.1f}s")
        else:
            # 首件或暂无有效样本时, 节拍显示占位符
            self.kpi_cycle_value.setText("--")

    @staticmethod
    def _set_pixmap_from_ndarray(label: ClickableLabel, ndata, is_bgr: bool = False):
        """将 numpy ndarray 转为 QPixmap 并设置到 QLabel (按比例缩放，不撑大布局)"""
        h, w, ch = ndata.shape
        if is_bgr:
            import cv2
            ndata = cv2.cvtColor(ndata, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        q_img = QImage(ndata.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        label.set_image(pix)

    # ================================================================
    #  定时器
    # ================================================================

    def _init_timers(self):
        # UI 数据刷新 (200ms)
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._update_ui_data)
        self.ui_timer.start(1000)

        # 时钟 (1000ms)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self._update_clock()
        self.clock_timer.start(1000)

    def _update_ui_data(self):
        """
        定时刷新 UI 数据 (200ms)
        ═══════════════════════════════════════════════════════════
        数据来源计划 (待 Controller 接入后启用):

            字段                      来源
        ───────────────────────────────────────────────────────
        末端坐标 X/Y/Z/R       self.controller.last_axis_status  (list[4])
        电机关节 J1~J4         self.controller.last_joint_status (list[4])
        当前动作文本            self.controller.current_action_msg (str)
        工艺步骤 index          self.controller.current_process_step (int 0-4)
        PLC 连接状态            self.controller.plc.is_connected (bool)
        相机连接状态            self.controller.vision_service.device.is_alive() (bool)
        产品型号                self.controller.cfg_manager.get("product_config.current_model", "")
        KPI 数据                从 stats.json 读取 (由 Controller 在每个 Cycle 结束后写入)
        相机画面 RGB/深度       通过 Controller.sig_camera_frame (pyqtSignal) 发射
                                _on_camera_frame 中 ndarray → QPixmap → setPixmap (已实现)

        使用示例:
            if self.controller is None:
                return  # 独立测试模式，只显示占位数据
            axis = self.controller.last_axis_status
            if axis and len(axis) == 4:
                self.coord_labels["X"].setText(f"{axis[0]:.1f}")
                self.coord_labels["Y"].setText(f"{axis[1]:.1f}")
                self.coord_labels["Z"].setText(f"{axis[2]:.1f}")
                self.coord_labels["R"].setText(f"{axis[3]:.1f}\u00b0")
        ═══════════════════════════════════════════════════════════
        """
        if self.controller is None:
            return  # 独立测试模式：无 Controller，仅占位
        # --- 以下为数据接入模板 (注释状态) ---
        axis = self.controller.last_axis_status
        if axis and len(axis) == 4:
            self.coord_labels["X"].setText(f"{axis[0]:.2f}   ")
            self.coord_labels["Y"].setText(f"{axis[1]:.2f}  ")
            self.coord_labels["Z"].setText(f"{axis[2]:.2f}   ")
            self.coord_labels["R"].setText(f"{axis[3]:.2f}\u00b0")

        joints = self.controller.last_joint_status
        if joints and len(joints) == 4:
            self.joint_labels["J1"].setText(f"{joints[0]:.2f}\u00b0 ")
            self.joint_labels["J2"].setText(f"{joints[1]:.2f}\u00b0")
            self.joint_labels["J3"].setText(f"{joints[2]:.2f} mm")
            self.joint_labels["J4"].setText(f"{joints[3]:.2f}\u00b0")

        if hasattr(self.controller, "current_action_msg"):
            msg = self.controller.current_action_msg
            if msg:
                self.action_text.setText(f">>> {msg}")

        if hasattr(self.controller, "current_process_msg"):
            msg = self.controller.current_process_msg
            if msg:
                self.process_current_label.setText(f"当前：{msg}")

        current_model = self.controller.cfg_manager.get_current_product_model()
        if current_model:
            self.model_label.setText(f"当前型号：{current_model}")

        if hasattr(self.controller, "current_process_step"):
            self.set_process_step(self.controller.current_process_step)

        # 硬件状态
        plc_ok = self.controller.plc and self.controller.plc.is_connected
        self.plc_dot.set_by_bool(plc_ok)

        cam_ok = (self.controller.vision_service
                  and self.controller.vision_service.device
                  and self.controller.vision_service.device.is_alive())
        self.cam_dot.set_by_bool(cam_ok)
        pass

    def _update_clock(self):
        now = QDateTime.currentDateTime()
        self.clock_label.setText(now.toString("HH:mm:ss"))

    # ================================================================
    #  交互
    # ================================================================

    def _show_fullscreen(self, pixmap):
        """显示全屏图像查看器"""
        if pixmap and not pixmap.isNull():
            self._fullscreen_overlay = FullscreenOverlay(pixmap, self)

    def _open_config(self):
        """
        打开配置界面 (密码保护)
        ────────────────────────────────────────────────────────
        计划:
            1. PasswordDialog 验证用户名/密码
            2. 通过后创建 ConfigEditorUI(controller_instance=self.controller)
            3. 调用 config_window.show() 作为独立二级窗口
            4. ConfigEditorUI 与 MainHMI 共享同一个 Controller 实例
               (包括 PLC 连接、相机服务、配置管理器等)
        ────────────────────────────────────────────────────────
        """
        dialog = PasswordDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            from src.ui.config_window import ConfigEditorUI
        except ImportError:
            QMessageBox.critical(self, "错误", "无法加载配置界面模块 (config_window.py)")
            return

        if self._config_window is not None:
            try:
                self._config_window.close()
            except Exception:
                pass
            self._config_window = None

        self._config_window = ConfigEditorUI(controller_instance=self.controller)
        self._config_window.show()


# ======================================================================
#  独立测试入口
# ======================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 尝试加载更美观的字体 (失败不影响运行)
    QFontDatabase.addApplicationFont(":/fonts/NotoSansCJKsc-Regular.otf")
    app.setFont(QFont("Microsoft YaHei", 9))

    window = MainHMI(controller=None)
    # window.set_process_step(2)
    window.show()

    sys.exit(app.exec_())
