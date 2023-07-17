# -*- coding: utf-8 -*-
# @Time : 2023/7/15 09:11
# @Author : qianyong

import cv2
import numpy as np
import videostab

video_file = "chen1.mp4"
stab = videostab.VideoStab()
cap = cv2.VideoCapture(video_file)
ret, prev = cap.read()

fourcc = cv2.VideoWriter.fourcc(*"mp4v")
fps = 60
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("out.mp4", fourcc, fps, (w * 2, h))

while True:
    try:
        ret, curr = cap.read()
        if not ret:
            break

        curr2 = stab.stabilize(prev, curr)

        # 拼接图像，输出到视频文件
        image = np.concatenate((curr, curr2), axis=1)
        out.write(image)

    except Exception as e:
        print(e)
        break

cap.release()
out.release()
print('done')
