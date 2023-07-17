import math

import cv2
import numpy as np

#https://github.com/Lakshya-Kejriwal/Real-Time-Video-Stabilization/

Q1 = 0.004
R1 = 0.5


class VideoStab(object):
    def __init__(self):
        self.k = 1
        self.errscaleX = 1
        self.errscaleY = 1
        self.errthetha = 1
        self.errtransX = 1
        self.errtransY = 1
        self.Q_scaleX = Q1
        self.Q_scaleY = Q1
        self.Q_thetha = Q1
        self.Q_transX = Q1
        self.Q_transY = Q1
        self.R_scaleX = R1
        self.R_scaleY = R1
        self.R_thetha = R1
        self.R_transX = R1
        self.R_transY = R1
        self.sum_scaleX = 0
        self.sum_scaleY = 0
        self.sum_thetha = 0
        self.sum_transX = 0
        self.sum_transY = 0
        self.scaleX = 0
        self.scaleY = 0
        self.thetha = 0
        self.transX = 0
        self.transY = 0

    def stabilize(self, frame_1, frame_2):
        frame1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        features1 = cv2.goodFeaturesToTrack(frame1, mask=None, **feature_params)
        features2, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, features1, None, **lk_params)

        goodFeatures1 = features1[st == 1]
        goodFeatures2 = features2[st == 1]

        T = cv2.estimateAffinePartial2D(goodFeatures1, goodFeatures2, False)
        dx = T[0][0, 2]
        dy = T[0][1, 2]
        da = math.atan2(T[0][1, 0], T[0][0, 0])
        ds_x = T[0][0, 0]/math.cos(da)
        ds_y = T[0][1, 1]/math.cos(da)
        T = np.matrix([[math.cos(da), -math.sin(da), dx], [math.sin(da), math.cos(da), dy]])
        smoothedFrame = cv2.warpAffine(frame_1, T, (len(frame_1[0]), len(frame_1)))
        return smoothedFrame

        sx = ds_x
        sy = ds_y

        self.sum_transX += dx
        self.sum_transY += dy
        self.sum_thetha += da
        self.sum_scaleX += ds_x
        self.sum_scaleY += ds_y

        if self.k == 1:
            self.k += 1
        else:
            # kalman filter
            # 输入：scaleX, scaleY, thetha, tranX, transY
            # 输出：scaleX, scaleY, thetha, tranxS, transY, errscaleX, errscaleY, errthetha, errtransX, errtransY
            self.Kalman_Filter()

        diff_scaleX = self.scaleX - self.sum_scaleX
        diff_scaleY = self.scaleY - self.sum_scaleY
        diff_transX = self.transX - self.sum_transX
        diff_transY = self.transY - self.sum_transY
        diff_thetha = self.thetha - self.sum_thetha

        ds_x = ds_x + diff_scaleX
        ds_y = ds_y + diff_scaleY
        dx = dx + diff_transX
        dy = dy + diff_transY
        da = da + diff_thetha

        T = np.matrix([[sx * math.cos(da), sx * -math.sin(da), dx], [sy * math.sin(da), sy * math.cos(da), dy]])
        T = np.matrix([[math.cos(da), -math.sin(da), dx], [math.sin(da), math.cos(da), dy]])
        smoothedFrame = cv2.warpAffine(frame_1, T, (len(frame_1[0]), len(frame_1)))
        return smoothedFrame

    def Kalman_Filter(self):
        return
        frame_1_scaleX = self.scaleX
        frame_1_scaleY = self.scaleY
        frame_1_thetha = self.thetha
        frame_1_transX = self.transX
        frame_1_transY = self.transY

        frame_1_errscaleX = self.errscaleX + self.Q_scaleX
        frame_1_errscaleY = self.errscaleY + self.Q_scaleY
        frame_1_errthetha = self.errthetha + self.Q_thetha
        frame_1_errtransX = self.errtransX + self.Q_transX
        frame_1_errtransY = self.errtransY + self.Q_transY

        gain_scaleX = frame_1_errscaleX / (frame_1_errscaleX + self.R_scaleX)
        gain_scaleY = frame_1_errscaleY / (frame_1_errscaleY + self.R_scaleY)
        gain_thetha = frame_1_errthetha / (frame_1_errthetha + self.R_thetha)
        gain_transX = frame_1_errtransX / (frame_1_errtransX + self.R_transX)
        gain_transY = frame_1_errtransY / (frame_1_errtransY + self.R_transY)

        self.scaleX = frame_1_scaleX + gain_scaleX * (self.sum_scaleX - frame_1_scaleX)
        self.scaleY = frame_1_scaleY + gain_scaleY * (self.sum_scaleY - frame_1_scaleY)
        self.thetha = frame_1_thetha + gain_thetha * (self.sum_thetha - frame_1_thetha)
        self.transX = frame_1_transX + gain_transX * (self.sum_transX - frame_1_transX)
        self.transY = frame_1_transY + gain_transY * (self.sum_transY - frame_1_transY)

        self.errscaleX = ( 1 - gain_scaleX ) * frame_1_errscaleX
        self.errscaleY = ( 1 - gain_scaleY ) * frame_1_errscaleX
        self.errthetha = ( 1 - gain_thetha ) * frame_1_errthetha
        self.errtransX = ( 1 - gain_transX ) * frame_1_errtransX
        self.errtransY = ( 1 - gain_transY ) * frame_1_errtransY
