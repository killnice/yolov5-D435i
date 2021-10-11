# yolov5-D435i
using yolov5 and realsense D435i

This repository contains a moded version of PyTorch YOLOv5 (https://github.com/ultralytics/yolov5)

本项目基于yolov5(https://github.com/ultralytics/yolov5)


将D435深度相机和yolov5结合到一起，在识别物体的同时，还能测到物体相对与相机的距离

硬件准备：

![image](https://github.com/killnice/yolov5-D435i/blob/main/realsense.png)


D435i是一个搭载IMU（惯性测量单元，采用的博世BMI055）的深度相机，D435i的2000万像素RGB摄像头和3D传感器可以30帧/秒的速度提供分辨率高达1280 × 720，或者以90帧/秒的速度提供848 × 480的较低分辨率。该摄像头为全局快门，可以处理快速移动物体，室内室外皆可操作。深度距离在0.1 m~10 m之间

计算机 win10 or ubuntu 最好有nvidia显卡

软件准备：

使用pip安装所需的包，进入本工程目录下

pip install -r requirements.txt

pip install pyrealsense2

# 程序运行

命令行cd 进入工程文件夹下

python main_debug.py

注意： 第一次运行程序程序会从云端下载yolov5的pt文件，大约140MB+ 

运行效果：

![image](https://github.com/killnice/yolov5-D435i/blob/main/test.gif)

