# import torch
import time
import numpy as np
import cv2
def dectshow(org_img, boxs):
    img = org_img.copy()
    for box in boxs:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('dec_img', img)
temp = np.load('temp.npy',allow_pickle=True)

print(type(temp[0][6]))
img = np.zeros((480,640,3),dtype=np.float)
for tex in temp:
    print(tex.shape)
    cv2.rectangle(img, (tex[0], tex[1]), (tex[2], tex[3]), (0, 255, 0), 2)

# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
# # Image
# img = 'https://ultralytics.com/images/zidane.jpg'
#
# # Inference
# time_x = time.time()
# results = model(img)
# print(time.time() - time_x)
# print(results)
# results.show()