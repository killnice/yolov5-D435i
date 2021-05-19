# yolov5-D435i
Introduction
using yolov5 and realsense D435i
This repository contains a moded version of PyTorch YOLOv5 (https://github.com/ultralytics/yolov5)
本项目基于yolov5（(https://github.com/ultralytics/yolov5)）
将D435深度相机和yolov5结合到一起，在识别物体的同时，还能测到物体相对与相机的距离

硬件准备：

![image](https://github.com/killnice/yolov5-D435i/blob/main/realsense.png)
D435i是一个搭载IMU（惯性测量单元，采用的博世BMI055）的深度相机，D435i的2000万像素RGB摄像头和3D传感器可以30帧/秒的速度提供分辨率高达1280 × 720，或者以90帧/秒的速度提供848 × 480的较低分辨率。该摄像头为全局快门，可以处理快速移动物体，室内室外皆可操作。深度距离在0.1 m~10 m之间

计算机 win10 or ubuntu 最好有nvidia显卡

软件准备：

使用pip安装所需的包
# pip install -r requirements.txt

# base ----------------------------------------
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
Pillow
PyYAML>=5.3.1
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.41.0

# logging -------------------------------------
tensorboard>=2.4.1
# wandb

# plotting ------------------------------------
seaborn>=0.11.0
pandas

# export --------------------------------------
# coremltools>=4.1
# onnx>=1.9.0
# scikit-learn==0.19.2  # for coreml quantization

# extras --------------------------------------
thop  # FLOPS computation
pycocotools>=2.0  # COCO mAP


