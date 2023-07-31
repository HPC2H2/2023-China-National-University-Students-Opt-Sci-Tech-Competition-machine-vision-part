
import cv2
import numpy as np
import time
from tkinter import *
from PIL import Image, ImageTk
import pyperclip  # 导入用于复制和粘贴的库

# 定义全局变量，用于储存LAB颜色空间阈值
l_min_g, a_min_g, b_min_g, l_max_g, a_max_g, b_max_g = 0, 0, 0, 255, 255, 255

class App:
    def __init__(self, window, window_title, video_source=0):
        # 初始化应用，设置窗口，窗口标题，视频源
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.ok = False

        # 打开视频源
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # 创建两个画布，用于显示原始视频和处理后的视频
        self.canvas1 = Canvas(window, width=480, height=360)
        self.canvas1.pack(side=LEFT)
        self.canvas2 = Canvas(window, width=480, height=360)
        self.canvas2.pack(side=LEFT)

        # 创建"Snapshot"按钮，用于拍摄快照
        self.btn_snapshot = Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=CENTER, expand=True)

        # 创建滑块，用于实时调整LAB颜色空间阈值
        self.l_min_slider = Scale(window, from_=0, to=255, orient=HORIZONTAL, label="L_min", length=480)
        self.l_min_slider.pack()

        self.a_min_slider = Scale(window, from_=0, to=255, orient=HORIZONTAL, label="A_min", length=480)
        self.a_min_slider.pack()

        self.b_min_slider = Scale(window, from_=0, to=255, orient=HORIZONTAL, label="B_min", length=480)
        self.b_min_slider.pack()

        self.l_max_slider = Scale(window, from_=0, to=255, orient=HORIZONTAL, label="L_max", length=480)
        self.l_max_slider.pack()

        self.a_max_slider = Scale(window, from_=0, to=255, orient=HORIZONTAL, label="A_max", length=480)
        self.a_max_slider.pack()

        self.b_max_slider = Scale(window, from_=0, to=255, orient=HORIZONTAL, label="B_max", length=480)
        self.b_max_slider.pack()

        # 创建"Copy LAB Values"按钮，用于复制当前的LAB颜色空间阈值
        self.btn_copy = Button(window, text="Copy LAB Values", command=self.copy_values)
        self.btn_copy.pack()

        # 定义刷新频率
        self.delay = 10
        self.update()

        # 启动主循环
        self.window.mainloop()

    def snapshot(self):
        # 如果视频已经打开，保存当前帧为图像文件
        if self.ok:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))

    def copy_values(self):
        # 获取滑块的值，然后将值格式化为字符串，并复制到剪切板
        l_min = self.l_min_slider.get()
        a_min = self.a_min_slider.get()
        b_min = self.b_min_slider.get()
        l_max = self.l_max_slider.get()
        a_max = self.a_max_slider.get()
        b_max = self.b_max_slider.get()

        values_str = f"({l_min}, {a_min}, {b_min}, {l_max}, {a_max}, {b_max})"
        pyperclip.copy(values_str)

    def update(self):
        # 读取当前帧，转换为RGB颜色空间，然后显示在画布上
        ret, self.frame = self.vid.read()
        if ret:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.photo1 = ImageTk.PhotoImage(image=Image.fromarray(self.frame).resize((480, 360)))
            self.canvas1.create_image(0, 0, image=self.photo1, anchor=NW)

            # 获取滑块的值，生成对应的颜色阈值，然后对当前帧进行阈值分割，并显示在第二个画布上
            l_min = self.l_min_slider.get()
            a_min = self.a_min_slider.get()
            b_min = self.b_min_slider.get()
            l_max = self.l_max_slider.get()
            a_max = self.a_max_slider.get()
            b_max = self.b_max_slider.get()

            lower_bound = np.array([l_min, a_min, b_min])
            upper_bound = np.array([l_max, a_max, b_max])
            mask = cv2.inRange(cv2.cvtColor(self.frame, cv2.COLOR_RGB2Lab), lower_bound, upper_bound)

            self.photo2 = ImageTk.PhotoImage(image=Image.fromarray(mask).resize((480, 360)))
            self.canvas2.create_image(0, 0, image=self.photo2, anchor=NW)

            self.ok = True

        # 延时后再次刷新画布
        self.window.after(self.delay, self.update)

# 创建一个App对象，并指定视频源为2
App(Tk(), "LAB Thresholding", video_source=2)