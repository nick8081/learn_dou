import cv2
import numpy as np

#https://github.com/Lakshya-Kejriwal/Real-Time-Video-Stabilization/

Q1 = 0.004
R1 = 0.5


class VideoStab(object):
    smoothedMat = 1.0 * np.zeros(2, 3)
