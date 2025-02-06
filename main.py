"""
This is an application designed for particle tracking and Mean Squared Displacement (MSD) calculation 
to determine the diffusion coefficient. It is used for the teaching of "Measurement of Avogadro's constant via diffusion coefficient" in the course of "物理实验(下)" of Fudan University.

Features include:
- Particle tracking in video or sequential frames
- MSD calculation and statistical error estimation
- Diffusion coefficient calculation
- Handling of average drift

Author: xyc
Email: 22307110070@m.fudan.edu.cn
Date: 2025-2-4

The GUI part is based on PyQt, and particle tracking and diffusion calculations are powered by the `trackpy` library(https://github.com/soft-matter/trackpy).
"""

# Import the necessary library
import trackpy as tp
from scipy.optimize import curve_fit
import sys
import cv2
import trackpy as tp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QFileDialog, QLabel, QLineEdit, QTabWidget, QFormLayout,QCheckBox,QTextEdit,QFileDialog, QProgressDialog, QMessageBox,QDialog,QProgressBar,QScrollArea
from PyQt5.QtCore import Qt,QObject, pyqtSignal
import logging
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from concurrent.futures import ThreadPoolExecutor, Future


LOGLEVEL="DEBUGGER"
GlobalTrajectory=None
GlobalFrameSource=None

def imagej2tpy(data):
    data.rename(columns={'TRACK_ID':"particle" }, inplace=True)
    data.rename(columns={'FRAME':"frame" }, inplace=True)
    data.rename(columns={'POSITION_X':"x" }, inplace=True)
    data.rename(columns={'POSITION_Y':"y" }, inplace=True)
    data=data.drop([0,1,2])
    data["frame"]=data["frame"].astype(float)
    data["x"]=data["x"].astype(float)
    data["y"]=data["y"].astype(float)
    data["particle"]=data["particle"].astype(float)
    #更改列名称及数据类型
    data=data.sort_values(by='frame')
    #按帧序号排列
    return data


# 自定义日志处理器，用于将日志信息传递到弹窗
class QtLogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)  # 定义一个信号，用于传递日志消息

    def __init__(self):
        super().__init__()
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)  # 获取日志消息
        self.log_signal.emit(msg)  # 通过信号发送消息


# 弹窗类
class BatchProgressDialog(QDialog):
    error_signal = pyqtSignal(str)
    def __init__(self,batchname="Tracking Process",parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tracking Process")
        self.setModal(True)  # 设置为模态窗口
        self.setGeometry(100, 100, 400, 300)
        # 初始化线程池
        ncpu=os.cpu_count()
        nth=max([ncpu-4,1,ncpu//2])
        self.executor = ThreadPoolExecutor(max_workers=nth)
        self.future = None
        self.error_signal.connect(self.show_error)
        # 布局
        layout = QVBoxLayout()
        # 日志显示区域
        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)  # 设置为只读
        layout.addWidget(self.log_display)
        self.setLayout(layout)
        # 初始化状态
        self.is_cancelled = False

    def update_log(self, msg):
        """更新日志显示"""
        self.log_display.append(msg)
    def show_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_msg}")


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
        self.setWindowTitle("BrownianAnalysis")
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
        self.if_load_frame=False

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
        self.invert_input = QCheckBox("")
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

        slider_layout = QHBoxLayout()
        # 帧滑动条
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.update_frame)
         # 滑动条值显示标签
        self.slider_label = QLabel(f"Frame: {self.frame_slider.value()}")
        self.slider_label.setAlignment(Qt.AlignCenter)
        # 将 QLabel 和 QSlider 添加到水平布局
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.frame_slider)

        # 添加到主垂直布局
        right_layout.addLayout(slider_layout)


        

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
            bg=15
            if len(imagefolder)>bg:
                self.img_label.setText("Selected path:..."+imagefolder[-bg::])
            else:
                self.img_label.setText("Selected path:"+imagefolder)
            global GlobalFrameSource
            GlobalFrameSource=self.file_path[bg:-1:]
            # 如果成功加载图片，更新 framelist
            self.framelist = np.array(image_sequence)
            
            self.is_video=False
            self.if_load_frame=True
            self.nframes=len(self.framelist)
            self.frame_slider.setMaximum(self.nframes-1)
            step=min(max(self.nframes//1000,1),10)
            self.log_message(f"step:{step}","DEBUGGER")
            self.frame_slider.setSingleStep(step)
            if self.nframes>2000:
                self.frame_slider.setTickInterval(step)
            self.log_message(f"Loaded image sequence from: {imagefolder},Total frames: {self.nframes}", "INFO")
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
        try:
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
            bg=max(-10,-len(self.file_path))
            self.log_message(f"Loaded video: {self.file_path}, fps: {fps}", "INFO")
            bg=15
            if len(self.file_path)>bg:
                self.file_label.setText("Selected File:..."+self.file_path[-bg::])
            else:
                self.file_label.setText("Selected File:"+self.file_path)
            global GlobalFrameSource
            GlobalFrameSource=self.file_path[bg::]
            self.is_video=True
            self.if_load_frame=True
            self.update_frame()
            self.nframes=len(self.framelist)
            self.frame_slider.setMaximum(self.nframes-1)
            step=min(max(self.nframes//1000,1),10)
            self.log_message(f"step:{step}","DEBUGGER")
            self.frame_slider.setSingleStep(step)
            if self.nframes>2000:
                self.frame_slider.setTickInterval(step)
            self.log_message(f"Loaded image sequence from: {self.file_path},Total frames: {self.nframes}", "INFO")
        except Exception as e:
            self.log_message(f"Failed to load video file with error: {str(e)}", "INFO")
            self.framelist = np.array([])


        
    def update_frame(self):
        try:
            frame_idx = self.frame_slider.value()
            if not self.if_load_frame:
                self.log_message("No frame loaded","INFO")
                return
            self.current_frame = self.framelist[frame_idx]
            self.plot_frame()
            self.slider_label.setText(f"Frame: {frame_idx}")
        except Exception as e:
            self.log_message(f"Fail to plot frame {frame_idx} with error {str(e)}","DEBUGGER")


    def plot_frame(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.current_frame, cmap='gray',aspect='equal')
        self.canvas.draw()

    def preview_particles(self):
        if not self.if_load_frame:
            self.log_message("No frame loaded","INFO")
            return
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
            self.log_message(f"Fail to locate particle as error {str(e)}","INFO")

    def track_particles(self):
        """启动 batch 处理"""
        if not self.if_load_frame:
            self.log_message("No frame loaded","INFO")
            return

        # 创建弹窗
        self.progress_dialog = BatchProgressDialog(self)
        self.progress_dialog.show()

        # 设置日志处理器
        self.log_handler = QtLogHandler()
        self.log_handler.setFormatter(logging.Formatter('%(message)s'))
        self.log_handler.log_signal.connect(self.progress_dialog.update_log)  # 连接信号到弹窗
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Creating tracking process...")
        # 在后台运行 batch
        self.progress_dialog.future = self.progress_dialog.executor.submit(self._track_particles)
        




    def _track_particles(self):
        self.provide_value()
        global GlobalTrajectory
        try:
            radius = get_radius(float(self.diameter_input.text()))
            invert = self.invert_input.isChecked()
            minmass = int(self.minmass_input.text())
            separation = int(self.separation_input.text())
            searchrange=int(self.searchrange_input.text())
            memory=int(self.memory_input.text())

            # 执行粒子跟踪
            t = tp.batch(self.framelist, radius, invert=invert, minmass=minmass, separation=separation,processes="auto")
            self.trajectories=tp.link(t,search_range=searchrange, memory=memory)
            # 关闭加载弹窗
            self.plot_trajectories()
            GlobalTrajectory=self.trajectories
        except Exception as e:
            # 关闭加载弹窗
            self.log_message(f"Fail to track as the error:{str(e)}","INFO")
            # 通过信号传递错误信息到主线程
            self.progress_dialog.error_signal.emit(str(e))
        finally:
            # 清理日志处理器
            logging.getLogger().removeHandler(self.log_handler)

    def closeEvent(self, event):
        """关闭主窗口时终止后台任务"""
        if self.progress_dialog.future and not self.progress_dialog.future.done():
            self.progress_dialog.future.cancel()
        self.progress_dialog.executor.shutdown(wait=False)
        event.accept()


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
                self.log_message(f"Trajectorise has be saved to path:{file_path}","INFO")
            else:
                self.log_message("Invalid file path")
        else:
            self.log_message("empty trajactories","INFO")
            

        
                

class MSDTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.if_load_data=False

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

        # CSV from Track
        self.send_label = QLabel("No file selected")
        self.send_button = QPushButton("Send CSV from Track Page")
        self.send_button.clicked.connect(self.send_csv)
        left_layout.addWidget(self.send_label)
        left_layout.addWidget(self.send_button)

        # 参数输入
        form_layout = QFormLayout()
        self.micron_per_pixel_input = QLineEdit("1.0")
        self.fps_input = QLineEdit("1.0")
        self.filtersubs_input = QLineEdit("0")
        self.drift_input = QCheckBox("")
        self.smoothwindow_input = QLineEdit("100")
        self.errorthreahold_input = QLineEdit("0.1")

        form_layout.addRow("Micron per Pixel:", self.micron_per_pixel_input)
        form_layout.addRow("FPS:", self.fps_input)
        form_layout.addRow("Filter Stubs:", self.filtersubs_input)
        form_layout.addRow("Subtract Drift:", self.drift_input)
        form_layout.addRow("Smooth Window:", self.smoothwindow_input)
        form_layout.addRow("Error Threshold:", self.errorthreahold_input)
        left_layout.addLayout(form_layout)

        # 计算 MSD 按钮
        self.calculate_button = QPushButton("Calculate MSD")
        self.calculate_button.clicked.connect(self.calculate_msd)
        left_layout.addWidget(self.calculate_button)

        #导出msd按钮
        self.export_button = QPushButton("Export MSD to CSV")
        self.export_button.clicked.connect(self.export_csv)
        left_layout.addWidget(self.export_button)

        #日志显示区
        # self.log_area = QTextEdit()
        # self.log_area.setReadOnly(True)  # 设置为只读
        # self.log_area.setMaximumHeight(100)  # 设置日志显示区的高度
        # left_layout.addWidget(QLabel("Log:"))
        # left_layout.addWidget(self.log_area)
        self.log_label = QLabel("Log:")
        left_layout.addWidget(self.log_label)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        scroll = QScrollArea()
        scroll.setWidget(self.log_area)
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(200, 100)  # 设置最小尺寸

        left_layout.addWidget(scroll)

        left_panel.setLayout(left_layout)

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
            data = pd.read_csv(self.file_path)
            required_columns = ['x', 'y', 'frame', 'particle']
            if not all(col in data.columns for col in required_columns):
                try:
                    data=imagej2tpy(data)
                except:
                    self.log_message("Invalid CSV file format","INFO")
                    return
            bg=15
            if len(self.file_path)>bg:
                self.file_label.setText("Selected File:..."+self.file_path[-bg::])
            else:
                self.file_label.setText("Selected File:"+self.file_path)
                

            self.log_message(f"Loaded CSV file: {self.file_path}","INFO")
            self.trajectories=data
        self.if_load_data=True
    def send_csv(self):
        try:
            self.trajectories=GlobalTrajectory
            self.if_load_data=True
            if self.trajectories is not None:
                self.log_message("Received trajectories from track page","INFO")
                bg=10
                if len(GlobalFrameSource)>bg:
                    self.send_label.setText("Received trajectories from:..."+GlobalFrameSource[-bg::])
                else:
                    self.send_label.setText("Received trajectories from:"+GlobalFrameSource)
            else:
                self.log_message("No trajectories received from track page","INFO")
        except Exception as e:
            self.log_message(f"Fail to send csv from track page as error:{str(e)}","INFO")
    def export_csv(self):
        if hasattr(self, 'msd'):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if file_path:

                self.msd.to_csv(file_path, index=False)
                self.log_message(f"MSD has be saved to path:{file_path}","INFO")
            else:
                self.log_message("Invalid file path")
        else:
            self.log_message("empty MSD","INFO")
    def log_message(self, message,level="DEBUGER"):
        """向日志显示区添加日志信息"""
        if getloglevel(LOGLEVEL)>=getloglevel(level):
            self.log_area.append(message)  # 追加日志信息
            self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum()) 


    def calculate_msd(self):
        #TODO:
        #漂移的去头尾smooth
        #错误处理
        #修饰图注增加标题 
        if hasattr(self, 'trajectories'):
            try:
                mpp = float(self.micron_per_pixel_input.text())
                fps = float(self.fps_input.text())
                swindow=int(self.smoothwindow_input.text())
                if_drift=self.drift_input.isChecked()
                filter_stubs=min(0,int(self.filtersubs_input.text()))
                errorthreahold=float(self.errorthreahold_input.text())

                t1 = tp.filter_stubs(self.trajectories,filter_stubs)

                d = tp.compute_drift(t1,smoothing=swindow)

                if if_drift:
                    tm = tp.subtract_drift(t1.copy(), d)
                else:
                    tm=t1
                em = tp.emsd(tm,mpp=mpp,fps=fps,detail=True,max_lagtime=999999)
                errors=1/np.sqrt(em["N"])
                maxlagt=np.count_nonzero(1/np.sqrt(em["N"])<errorthreahold)
                self.figure.clear()
                ax1=self.figure.add_subplot(221)
                tp.plot_traj(tm,ax=ax1)
                ax1.set_title('Trajectories')
                ax2 = self.figure.add_subplot(222)
                ax2.plot(d.index,d["x"]*mpp,label="x")
                ax2.plot(d.index,d["y"]*mpp,label="y")
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Mean Drift')
                ax2.set_title('Mean Drift of x,y')
                ax2.legend()
                ax3 = self.figure.add_subplot(223)
                x,y=em["lagt"],em["msd"]
                x0=x.iloc[maxlagt]
                ymin=np.min(y)
                ymax=np.max(y+np.abs(errors*y))

                ### 箭头标注
            #     arrowy=160
            #     arrowx=20
            #     ax3.annotate(
            #     r'$\frac{\Delta \rho_k}{<\rho_k>}<10\%$', 
            #     fontsize=14,# 注释文本
            #     xy=(x0-arrowx, arrowy+2),            # 箭头指向的坐标
            #     xytext=(x0, arrowy),       # 注释文本的位置
            #     arrowprops=dict(
            #         arrowstyle="->",  # 箭头样式
            #         color="red",     # 箭头颜色
            #         lw=0           # 箭头宽度
            #     )
            # )
                ###
                
                ax3.plot([x0,x0],[ymin,ymax],c="r",linestyle="--")
                ax3.set_xlabel('Time lag/s')
                ax3.set_ylabel(r"$MSD/\mu m^2$")
                ax3.set_title('Mean Square Displacement')
                ax3.scatter(x, y,s=3,marker="s")
                ax3.errorbar(x, y, yerr=y*errors, fmt='s', markersize=2,ecolor='gray', capsize=0.05,alpha=0.1)
                ax4 = self.figure.add_subplot(224)
                def linear_func(x, k):
                    return k * x
                x=em["lagt"][:maxlagt]
                y=em["msd"][:maxlagt]
                errors=1/np.sqrt(em["N"])[:maxlagt]         
                # 进行带误差的拟合
                popt, pcov = curve_fit(linear_func, x, y,sigma=errors)
                # 提取拟合参数和误差
                slope = popt[0]
                slope_error = np.sqrt(pcov[0, 0])
                # 绘制数据
                ax4.scatter(x, y,s=3,marker="s")
                ax4.errorbar(x, y, yerr=y*errors, fmt='s', markersize=2,ecolor='gray', capsize=0.05,alpha=0.2)
                #绘制拟合线
                ax4.plot(x, slope * x,c="r")
                y_fit=slope * x 
                # 计算R²
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r = 1 - (ss_res / ss_tot)
                prc=getprc(slope_error)
                prcR=getprc(1-r**2)
                prcd=getprc(slope_error/4)
                ax4.set_ylabel(r"$MSD/\mathrm{\mu m^2}$",size=10)
                ax4.set_xlabel(r"$\Delta t/\mathrm{s}$",size=10)
                ax4.legend([r"$MSD-\Delta t$",r"fit with $y=mx$"])
                ax4.set_title('Mean Square Displacement Fit')
                
                xt=np.max(x)*0.4
                yt=np.min(y)+(np.max(y)-np.min(y))*0.1
                ax4.text(xt,yt,f"m={slope:.{prc+1}f}±{slope_error:.{prc+1}f} μm²/s\nR²={r**2:.{prcR}f}")
                self.canvas.draw()

             
                self.log_message(f"prc,prcd,prcR:{prc},{prcd},{prcR}","DEBUGGER")
                self.log_message("MSD has been calculated","INFO")
                self.log_message(f"Fit with y=m x:slope:{slope:.{prc+1}f}±{slope_error:.{prc+1}f} μm²/s,R²={r**2:.{prcR}f}","INFO")
                self.log_message(f"Diffusion coefficient: {slope/4:.{prcd+1}f} ± {slope_error/4:.{prcd+1}f} μm²/s","INFO")

                ###export msd
                self.msd=em.copy()
                self.msd['Relative Error'] = errors 
            except Exception as e:
                self.log_message(f"Fail to calculate MSD as error:{str(e)}","INFO")
        else:
            self.log_message("empty trajectories")

def getprc(dx):
    prc = int(-np.floor(np.log10(abs(dx))))
    return prc

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())