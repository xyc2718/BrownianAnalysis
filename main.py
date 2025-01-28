import sys
import cv2
import trackpy as tp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from brown import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QFileDialog, QLabel, QLineEdit, QTabWidget, QFormLayout,QCheckBox,QTextEdit,QFileDialog, QProgressDialog, QMessageBox
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

def get_radius(diameter):
    r=int(np.floor(diameter/2))
    if r%2==0:
        return r+1
    else:
        return r
    


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
        self.diameter_input = QLineEdit("30")
        self.invert_input = QCheckBox("Invert")
        self.minmass_input = QLineEdit("10")
        self.separation_input = QLineEdit("30")
        self.diameter_input.textChanged.connect(self.defaut_trackvalue)
        self.searchrange_input=QLineEdit("80")
        self.memory_input=QLineEdit("0")

        form_layout.addRow("Diameter:", self.diameter_input)
        form_layout.addRow("Invert:", self.invert_input)
        form_layout.addRow("Minmass:", self.minmass_input)
        form_layout.addRow("Separation:", self.separation_input)
        form_layout.addRow("Search Range:",self.searchrange_input)
        form_layout.addRow("Memory:",self.memory_input)
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

        #histplot
        self.hfigure = Figure(figsize=(6,4))
        self.hcanvas = FigureCanvas(self.hfigure)
        left_layout.addWidget(self.hcanvas)

        left_panel.setLayout(left_layout)

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

    def provide_value(self):
        try:
            radius = get_radius(float(self.diameter_input.text()))
            minmass = int(self.minmass_input.text())
            separation = int(self.separation_input.text())
            m=int(self.memory_input.text())
            sr=int(self.searchrange_input.text())
        except:
            self.log_message("Error! Invalid input Value!")
            self.separation_input.setText(f"10")
            self.minmass_input.setText(f"10")
            self.diameter_input.setText(f"10")
            self.searchrange_input.setText(f"20")
            self.memory_input.setText(f"0")

        
            

            
            
            
    def defaut_trackvalue(self):
        try:
            radius = get_radius(float(self.diameter_input.text()))
            separation = 2*radius
            self.separation_input.setText(f"{separation}")
        except:
            self.provide_value()

        

    def log_message(self, message,level="DEBUGER"):
        """向日志显示区添加日志信息"""
        if getloglevel(LOGLEVEL)>=getloglevel(level):
            self.log_area.append(message)  # 追加日志信息
            self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum()) 



    def load_imagesequence(self):
        # 弹出文件夹选择对话框
        imagefolder = QFileDialog.getExistingDirectory(self, "Open Image Sequence", "")
        if not imagefolder:  # 如果用户取消选择，直接返回
            return
        try:
            # 获取文件夹中的所有文件
            files = os.listdir(imagefolder)
            # 过滤出支持的图片文件（支持.jpg和.bmp格式）
            image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.bmp')]
            # 按文件名中的数字排序
            image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            
            # 初始化一个空列表，用于存储图片数组
            image_sequence = []
            total_files = len(image_files)
            
            if total_files == 0:
                self.log_message(f"No valid image files found in folder: {imagefolder}", "INFO")
                self.framelist = np.array([])
                return

            # 初始化进度条
            progress_dialog = QProgressDialog("Loading image sequence...", "Cancel", 0, total_files, self)
            progress_dialog.setWindowTitle("Loading Progress")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()

            # 遍历排序后的图片文件并加载
            for i, image_file in enumerate(image_files):
                if progress_dialog.wasCanceled():  # 检查用户是否取消
                    self.log_message("Image sequence loading canceled by the user.", "INFO")
                    break
                # 构建完整的文件路径
                image_path = os.path.join(imagefolder, image_file)
                # 打开图片并转换为 np.array
                image = Image.open(image_path)
                gray_image = image.convert("L")
                image_array = np.array(gray_image)
                # 将图片数组添加到列表中
                image_sequence.append(image_array)

                # 更新进度条
                progress_dialog.setValue(i + 1)

            # 关闭进度条
            progress_dialog.close()
            self.file_path=image_path
            self.img_label.setText("Selected file: " + self.file_path[-10:-1:])
            # 如果成功加载图片，更新 framelist
            self.framelist = np.array(image_sequence)
            self.log_message(f"Loaded image sequence from: {imagefolder}", "INFO")
            self.is_video=False
            self.update_frame()

        except Exception as e:
            # 如果加载过程中出错，记录日志
            self.log_message(f"Failed to load image sequence with error: {str(e)}", "INFO")
            self.framelist = np.array([])

    
        

    def load_video(self):
        # 打开文件选择对话框
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")
        if not self.file_path:
            return  # 如果用户取消了文件选择，直接返回
        frames = []
        video = cv2.VideoCapture(self.file_path)

        # 确保视频文件成功打开
        if not video.isOpened():
            self.log_message(f"Failed to load video file: {self.file_path}", "INFO")
            self.framelist = np.array([])
            return

        # 获取视频总帧数和帧率
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        # 初始化进度条
        progress_dialog = QProgressDialog("Loading video...", "Cancel", 0, total_frames, self)
        progress_dialog.setWindowTitle("Loading Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()

        frame_count = 1  # 从1开始计数
        while True:
            # 检查是否取消
            if progress_dialog.wasCanceled():
                self.log_message("Video loading canceled by the user.", "INFO")
                break

            # 读取当前帧
            ret, frame = video.read()
            if not ret or frame_count > self.maxframe:
                break

            # 将帧转换为灰度图并添加到帧列表
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(np.array(gray_image))

            # 更新进度条
            progress_dialog.setValue(frame_count)
            frame_count += 1

        # 完成加载后处理
        progress_dialog.close()
        self.framelist = np.array(frames)
        self.log_message(f"Loaded video: {self.file_path}, fps: {fps}", "INFO")
        self.file_label.setText("Selected file:" + self.file_path[-10:-1:])
        self.is_video=True
        self.update_frame()

            

    def update_frame(self):
        try:
            frame_idx = self.frame_slider.value()
            self.current_frame = self.framelist[frame_idx]
            self.plot_frame()
        except Exception as e:
            self.log_message(f"Fail to plot frame {frame_idx} with error {str(e)}")


    def plot_frame(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.current_frame, cmap='gray',aspect='equal')
        self.canvas.draw()

    def preview_particles(self):
        self.provide_value()
        try:
            radius = get_radius(float(self.diameter_input.text()))
            invert = self.invert_input.isChecked()
            minmass = int(self.minmass_input.text())
            separation = int(self.separation_input.text())
            f = tp.locate(self.current_frame, radius, invert=invert, minmass=minmass, separation=separation)
            self.log_message(f"{radius},{invert},{minmass},{separation}","DEBUGGER")
            self.plot_frame()
            ax = self.figure.gca()
            tp.annotate(f, self.current_frame, plot_style={'markersize': radius, 'markeredgewidth': 2, 'markeredgecolor': 'r', 'markerfacecolor': 'None'}, ax=ax)
            self.canvas.draw()
            self.hfigure.clear()
            ax = self.hfigure.gca()
            ax.hist(f['mass'], bins=80)
            ax.set(xlabel='mass', ylabel='count')
            self.hcanvas.draw()
        except Exception as e:
            self.load_imagesequence(f"Fail to locate particle as error {str(e)}")


    def track_particles(self):
        self.provide_value()
        # 创建并显示加载弹窗
        progress_dialog = QProgressDialog("Tracking particles...", None, 0, 0, self)
        progress_dialog.setWindowTitle("Loading")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setCancelButton(None)  # 禁用取消按钮
        progress_dialog.show()
        try:
            radius = get_radius(float(self.diameter_input.text()))
            invert = self.invert_input.isChecked()
            minmass = int(self.minmass_input.text())
            separation = int(self.separation_input.text())
            searchrange=int(self.searchrange_input.text())
            memory=int(self.memory_input.text())
            # 执行粒子跟踪
            t = tp.batch(self.framelist, radius, invert=invert, minmass=minmass, separation=separation)
            self.trajectories=tp.link(t,search_range=searchrange, memory=memory)
            # 关闭加载弹窗
            self.plot_trajectories()
            progress_dialog.close()
        except Exception as e:
            # 关闭加载弹窗
            progress_dialog.close()
            self.load_imagesequence(f"Fail to track as the error:{str(e)}","INFO")
            # 显示错误信息
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def plot_trajectories(self):
            t=self.trajectories
            ax = self.figure.gca()
            kk=self.frame_slider.value()
            t_f=t[t["frame"]==kk]
            x = t_f['x']
            y = t_f['y']
            particle_id = t_f['particle']
            color=['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan',
                'magenta', 'lime', 'teal', 'olive', 'maroon', 'navy', 'aquamarine', 'indigo', 'thistle', 'peru',
                'rosybrown', 'darkslategray', 'lightcoral', 'slateblue', 'firebrick', 'darkolivegreen', 'darkcyan', 'khaki', 'mediumvioletred']
            # 在图上标注颗粒位置和编号
            for i, (xi, yi, pid) in enumerate(zip(x, y, particle_id)):
                cid=np.mod(pid,25)
                xhis=(t[t["particle"]==pid])["x"]
                yhis=(t[t["particle"]==pid])["y"]
                fhis=max(np.int16((t[t["particle"]==pid])["frame"]))
                ax.plot(xi, yi, 'o', markersize=5, label=f'Particle {pid}',c=color[cid])
                ax.plot(xhis[:kk-fhis],yhis[:kk-fhis],c=color[cid],alpha=0.5)
                # plt.text(xi, yi, str(pid), fontsize=18, color='b')  
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