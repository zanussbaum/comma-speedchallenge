import cv2
import numpy as np
np.set_printoptions(suppress=True)
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform


def extract_features(img):
    extractor = cv2.ORB_create()
    feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                                    1000, qualityLevel=0.01, minDistance=7)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]

    kps, des = extractor.compute(img, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


def make_homogenous(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def normalize(Kinv, pts):
    # K 3 x 3, pts is n x 3
    # want output to be n x 2
    # K dot pts.T == 3 X N
    return np.dot(Kinv, make_homogenous(pts).T).T[:, :2]

def denormalize(K, pts):
    denormed = np.dot(K, np.array([pts[0], pts[1], 1.0]).T)
    return int(round(denormed[0])), int(round(denormed[1]))


def poseRt(R, t):
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    return Rt


def extract_pose(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, d, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1.0

    if np.linalg.det(Vt) < 0:
        Vt *= -1.0

    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)

    t = U[:, 2]
    if t[2] < 0:
        t *= 1

    return np.linalg.inv(poseRt(R, t))


def match_frames(frame1, frame2, K):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    points = []
    idx1, idx2 = [], []

    matches = matcher.knnMatch(frame1.des, frame2.des, k=2)
    # lowe's ratio test OpenCV: rb.gy/w5yrmb
    for m, n in matches:
        if m.distance < .75*n.distance:
            p1 = frame1.kps[m.queryIdx]
            p2 = frame2.kps[m.trainIdx]
            if m.distance < 32:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                points.append((p1, p2))

    assert len(points) >= 8, len(points)
    points = np.array(points)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    model, inliers = ransac((points[:, 0], points[:, 1]),
                            EssentialMatrixTransform,
                            min_samples=8, residual_threshold=0.001,
                            max_trials=100)
    points = points[inliers]

    Rt = extract_pose(model.params)
    print("Matches: %d -> %d -> %d -> %d" % (len(frame1.des),
          len(matches), len(inliers), sum(inliers)))

    return idx1[inliers], idx2[inliers], Rt


class Frame:
    def __init__(self, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(K)

        self.h, self.w = img.shape[:2]

        self.kps, self.des = extract_features(img)
        self.kps = normalize(self.Kinv, self.kps)
        self.pose = np.eye(4)
