import numpy as np
import cv2
import math

SMOOTHING_RADIUS = 30
HORIZONTAL_BORDER_CROP = 20
DISPLAY_CORNERS = False

video_file = "chen1.mp4"
cap = cv2.VideoCapture(video_file)

ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

fourcc = cv2.VideoWriter.fourcc(*"mp4v")
fps = 60
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
slice_w_offset = 910  # 宽-切片位置
if slice_w_offset > 0:
    w = w - slice_w_offset
    prev = prev[:, slice_w_offset:]
    prev_gray = prev_gray[:, slice_w_offset:]
out = cv2.VideoWriter("out.mp4", fourcc, fps, (w * 2, h))

def kalman_filter(dx, dy, da):
    kalman = cv2.KalmanFilter(2, 1, 0)
    kalman.transitionMatrix = np.array([[0.],[]])


a, x, y = 0.0, 0.0, 0.0
trajectory = []
def avg_move_filter(dx, dy, da):
    """移动平均滤波
    """
    global a, x, y, trajectory
    # 计算移动窗口内的轨迹。后续计算滤波时要用。
    x += dx
    y += dy
    a += da
    trajectory.append((x, y, a))
    if len(trajectory) >= SMOOTHING_RADIUS:
        trajectory = trajectory[:SMOOTHING_RADIUS]

    # 平移平均滤波，得到滤波系数。
    sx, sy, sa, ctr = 0.0, 0.0, 0.0, 0
    for i, value in enumerate(reversed(trajectory)):
        if i >= SMOOTHING_RADIUS:
            break
        tx, ty, ta = value
        sx += tx
        sy += ty
        sa += ta
        ctr += 1
    sx, sy, sa = sx / ctr, sy / ctr, sa / ctr

    # 之前得到的拐点仿射变换值，这里经过滤波处理，得到平滑值
    nx, ny, na = trajectory[-1]
    tx, ty, ta = dx + sx - nx, dy + sy - ny, da + sa - na
    return tx, ty, ta


while True:
    try:
        ret, curr = cap.read()
        if not ret:
            break
        curr = curr[:, slice_w_offset:]
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        previous_corner = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        current_corner, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, previous_corner, None, **lk_params)

        previous_corner2 = previous_corner[st == 1]
        current_corner2 = current_corner[st == 1]

        # 计算拐点仿射变换值（后面还要进行一次滤波）
        T = cv2.estimateAffinePartial2D(previous_corner2, current_corner2, False)
        dx = T[0][0, 2]
        dy = T[0][1, 2]
        da = math.atan2(T[0][1, 0], T[0][0, 0])

        tx, ty, ta = avg_move_filter(dx, dy, da)

        # 通过拐点放射变换值，计算得到warp系数，执行warp
        T = np.matrix([[math.cos(ta), -math.sin(ta), tx], [math.sin(ta), math.cos(ta), ty]])
        curr2 = cv2.warpAffine(curr, T, (len(curr[0]), len(curr)))

        # 当前帧切换成过去帧。便于下一步计算新的拐点及其移动。
        prev = curr.copy()
        prev_gray = curr_gray.copy()

        # curr叠加corners展示
        curro = curr.copy()
        if DISPLAY_CORNERS:  # 叠加显示了之后，右边的图有较大偏移，尚未知原因
            for x, y in current_corner2:
                cv2.circle(curro, (int(x), int(y)), 5, (0, 0, 255), -1)

        # 拼接图像，输出到视频文件
        image = np.concatenate((curro, curr2), axis=1)
        out.write(image)
    except Exception as e:
        break

cap.release()
out.release()
print('done')
