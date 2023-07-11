import cv2
import numpy as np
# https://zhuanlan.zhihu.com/p/250839967

cap = cv2.VideoCapture('a.mp4')
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter.fourcc(*'mp4v')
fps = 60
out = cv2.VideoWriter("video_out.mp4", fourcc, fps, (w, h))

# read first frame
_, prev = cap.read()
# convert frame to gray scale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32)
for i in range(n_frames-2):
    # detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)
    #read next frame
    success, curr = cap.read()
    if not success:
        break
    #convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    #calculate optional flow(i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    #snaity check
    assert prev_pts.shape == curr_pts.shape
    #filter only valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    #find transformation matrix
    m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    #extract translation
    dx = m[0][0, 2]
    dy = m[0][1, 2]
    #extract rotation angle
    da = np.arctan2(m[1][1,0], m[1][0,0])
    #store transformation
    transforms[i] = [dx,dy,da]
    #move to next frame
    prev_gray = curr_gray
    print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

#compute trajectory using cumulative sum of transformation
trajectory = np.cumsum(transforms, axis=0)

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # define the filter
    f = np.ones(window_size)/window_size
    #add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    #remove padding
    curve_smoothed=curve_smoothed[radius:-radius]
    return curve_smoothed

SMOOTHING_RADIUS = 5
def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

smoothed_trajectory = smooth(trajectory)

#calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
#calculate newer transformation array
transforms_smooth = transforms + difference

#第五步:将平滑的摄像机运动应用到帧中
def fixBorder(frame):
    s = frame.shape
    #scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

#reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#write n_frames-1 transformed frames
for i in range(n_frames-2):
    # read next frame
    success, frame = cap.read()
    if not success:
        break
    #extract transformations from the new transformation array
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    dz = transforms_smooth[i, 2]
    #reconstruct transformation matrix accordingly to new value
    m = np.zeros((2,3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy
    #apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (w, h))
    # fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized)
    #write the frame to the file
    frame_out = cv2.hconcat(frame, frame_stabilized)
    #if the image is too big, resize it
    if(frame_out.shape[1] > 1920):
        frame_out = cv2.resize(frame_out, ((frame_out.shape[1]/2, frame_out.shape[0]/2)))
    cv2.imshow("before and after", frame_out)
    cv2.waitKey(10)
    out.write(frame_out)


print('done')