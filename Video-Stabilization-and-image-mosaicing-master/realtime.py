import numpy as np
import cv2
import math

SMOOTHING_RADIUS = 30
HORIZONTAL_BORDER_CROP = 20

video_file = "chen1.mp4"
cap = cv2.VideoCapture(video_file)

ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
previous_to_current_transform = []

fourcc = cv2.VideoWriter.fourcc(*"mp4v")
fps = 60
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("out.mp4", fourcc, fps, (w * 2, h))

a, x, y = 0.0, 0.0, 0.0
trajectory = []

smoothed_trajectory = []

new_previous_to_current_transform = []


while True:
    try:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        previous_corner = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        current_corner, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, previous_corner, None, **lk_params)

        previous_corner2 = previous_corner[st == 1]
        current_corner2 = current_corner[st == 1]

        T = cv2.estimateAffinePartial2D(previous_corner2, current_corner2, False)
        dx = T[0][0, 2]
        dy = T[0][1, 2]
        da = math.atan2(T[0][1, 0], T[0][0, 0])

        previous_to_current_transform.append((dx, dy, da))

        x += dx
        y += dy
        a += da
        trajectory.append((x, y, a))

        # 平移平均滤波
        sx, sy, sa, ctr = 0.0, 0.0, 0.0, 0
        for i, value in enumerate(reversed(trajectory)):
            if i >= SMOOTHING_RADIUS:
                break
            tx, ty, ta = trajectory[i]
            sx += tx
            sy += ty
            sa += ta
            ctr += 1
        smoothed_trajectory.append((sx / ctr, sy / ctr, sa / ctr))
        sx, sy, sa = smoothed_trajectory[-1]

        nx, ny, na = trajectory[-1]
        new_previous_to_current_transform.append((dx + sx - nx, dy + sy - ny, da + sa - na))

        tx, ty, ta = new_previous_to_current_transform[-1]
        T = np.matrix([[math.cos(ta), -math.sin(ta), tx], [math.sin(ta), math.cos(ta), ty]])
        curr2 = cv2.warpAffine(curr, T, (len(curr[0]), len(curr)))
        image = np.concatenate((curr, curr2), axis=1)
        out.write(image)

        prev = curr.copy()
        prev_gray = curr_gray.copy()
    except Exception as e:
        break

cap.release()
out.release()
print('done')
