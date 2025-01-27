import sys
import cv2
import trackpy as tp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QFileDialog, QLabel, QLineEdit, QTabWidget, QFormLayout,QCheckBox,QTextEdit
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

LOGLEVEL="DEBUGGER"

def getloglevel(level):
    if level=="DEBUGGER":
        return 1
    elif level=="INFO":
        return 0
    else:
        return -1
    

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Particle Tracking and MSD Analysis")
        self.setGeometry(100, 100, 1200, 800)

        # Create tabs
        self.tabs = QTabWidget()
        self.trajectory_tab = TrajectoryTab()
        self.msd_tab = MSDTab()
        self.tabs.addTab(self.trajectory_tab, "轨迹识别")
        self.tabs.addTab(self.msd_tab, "轨迹处理")

        self.setCentralWidget(self.tabs)

class TrajectoryTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.maxframe=100000

    def initUI(self):
        # 使用 QHBoxLayout 实现左右布局
        layout = QHBoxLayout()

        # 左侧参数设置区域
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # 视频文件加载
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Load Video")
        self.file_button.clicked.connect(self.load_video)
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.file_button)

        
        # 图片序列输入
        self.img_label = QLabel("No path selected")
        self.img_button = QPushButton("Load Image Sequence")
        self.img_button.clicked.connect(self.load_imagesequence)
        left_layout.addWidget(self.img_label)
        left_layout.addWidget(self.img_button)

        # 参数输入
        form_layout = QFormLayout()
        self.radius_input = QLineEdit("5")
        self.invert_input = QCheckBox("Invert")
        self.minmass_input = QLineEdit("100")
        self.separation_input = QLineEdit("5")
        form_layout.addRow("Radius:", self.radius_input)
        form_layout.addRow("Invert:", self.invert_input)
        form_layout.addRow("Minmass:", self.minmass_input)
        form_layout.addRow("Separation:", self.separation_input)
        left_layout.addLayout(form_layout)

        # 预览按钮
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview_particles)
        left_layout.addWidget(self.preview_button)

        # 轨迹跟踪按钮
        self.track_button = QPushButton("Track")
        self.track_button.clicked.connect(self.track_particles)
        left_layout.addWidget(self.track_button)

        # 导出按钮
        self.export_button = QPushButton("Export to CSV")
        self.export_button.clicked.connect(self.export_csv)
        left_layout.addWidget(self.export_button)

        #日志显示区
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)  # 设置为只读
        self.log_area.setMaximumHeight(100)  # 设置日志显示区的高度
        left_layout.addWidget(QLabel("Log:"))
        left_layout.addWidget(self.log_area)

        left_panel.setLayout(left_layout)

        # 添加伸缩空间
        left_layout.addStretch()

        left_panel.setLayout(left_layout)

        # 右侧显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # 帧滑动条
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.update_frame)
        right_layout.addWidget(self.frame_slider)

        

        # Matplotlib 图像显示
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        right_panel.setLayout(right_layout)

        # 将左右面板添加到主布局
        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(right_panel, stretch=3)

        self.setLayout(layout)

    def log_message(self, message,level="DEBUGER"):
        """向日志显示区添加日志信息"""
        if getloglevel(LOGLEVEL)>=getloglevel(level):
            self.log_area.append(message)  # 追加日志信息
            self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum()) 
    def load_imagesequence(self):
        pass
        
    def load_video(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")

        if self.file_path:
                    # 打开视频文件
            frames=[]
            video = cv2.VideoCapture(self.file_path)
            self.log_message(f"Loading Video...")
            # 确保视频文件成功打开
            if not video.isOpened():
                self.log_message(f"Fail to load video file{self.file_path}","INFO")
                self.framelist=np.array([])
                return
            # 逐帧导出为图片序列
            frame_count = 1  # 从1开始命名
            fps = video.get(cv2.CAP_PROP_FPS)
            print(f"视频帧率: {fps}")
            while True:
                # 读取当前帧
                ret, frame = video.read()
                
                # 检查是否成功读取帧
                if not ret:
                    break
                if frame_count>self.maxframe:
                    break    
                frame_count += 1
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(np.array(gray_image))
            print(22)
            self.framelist=np.array(frames)
            self.log_message(f"load video: {self.file_path}","INFO")
            self.update_frame()
            

    def update_frame(self):
        try:
            frame_idx = self.frame_slider.value()
            self.current_frame = self.framelist[frame_idx]
            self.plot_frame()
        except:
            self.log_message(f"Fail to plot frame {frame_idx}")
            pass

    def plot_frame(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.current_frame, cmap='gray')
        self.canvas.draw()

    def preview_particles(self):
        radius = int(self.radius_input.text())
        invert = self.invert_input.isChecked()
        minmass = int(self.minmass_input.text())
        separation = int(self.separation_input.text())

        f = tp.locate(self.current_frame, radius, invert=invert, minmass=minmass, separation=separation)
        self.plot_frame()
        ax = self.figure.gca()
        tp.annotate(f, self.current_frame, plot_style={'markersize': radius, 'markeredgewidth': 2, 'markeredgecolor': 'r', 'markerfacecolor': 'None'}, ax=ax)
        self.canvas.draw()

    def track_particles(self):
        radius = int(self.radius_input.text())
        invert = self.invert_input.text().lower() == 'true'
        minmass = int(self.minmass_input.text())
        separation = int(self.separation_input.text())

        self.trajectories = tp.batch(self.frames, radius, invert=invert, minmass=minmass, separation=separation)
        self.plot_trajectories()

    def plot_trajectories(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        tp.plot_traj(self.trajectories, ax=ax)
        self.canvas.draw()

    def export_csv(self):
        if hasattr(self, 'trajectories'):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if file_path:
                self.trajectories.to_csv(file_path, index=False)

class MSDTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 使用 QHBoxLayout 实现左右布局
        layout = QHBoxLayout()

        # 左侧参数设置区域
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # CSV 文件加载
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Load CSV")
        self.file_button.clicked.connect(self.load_csv)
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.file_button)

        # 计算 MSD 按钮
        self.calculate_button = QPushButton("Calculate MSD")
        self.calculate_button.clicked.connect(self.calculate_msd)
        left_layout.addWidget(self.calculate_button)

        # 添加伸缩空间
        left_layout.addStretch()

        left_panel.setLayout(left_layout)

        # 右侧显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Matplotlib 图像显示
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        right_panel.setLayout(right_layout)

        # 将左右面板添加到主布局
        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(right_panel, stretch=3)

        self.setLayout(layout)

    def load_csv(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if self.file_path:
            self.file_label.setText(self.file_path)
            self.trajectories = pd.read_csv(self.file_path)

    def calculate_msd(self):
        if hasattr(self, 'trajectories'):
            msd = tp.emsd(self.trajectories, mpp=1, fps=1)
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(msd.index, msd, 'o-')
            ax.set_xlabel('Time lag')
            ax.set_ylabel('MSD')
            self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())