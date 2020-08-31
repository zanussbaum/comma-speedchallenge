import cv2
import numpy as np
np.set_printoptions(suppress=True)
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


def make_homogenous(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

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


class Extractor:
    def __init__(self, K):
        self.extractor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.Fx = K[0, 0]
        self.Fy = K[1, 1]

    def normalize(self, pts):
        # K 3 x 3, pts is n x 3
        # want output to be n x 2
        # K dot pts.T == 3 X N
        return np.dot(self.Kinv, make_homogenous(pts).T).T[:, :2]

    def denormalize(self, pts):
        denormed = np.dot(self.K, np.array([pts[0], pts[1], 1.0]).T)
        return int(round(denormed[0])), int(round(denormed[1]))

    def extract(self, img):
        # changing to gray scale
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                                        3000, qualityLevel=0.01, minDistance=3)

            # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.extractor.compute(img, kps)

        points = []
        if self.last is not None:
            matches = self.matcher.knnMatch(des, self.last['des'], k=2)
            # lowe's ratio test OpenCV: rb.gy/w5yrmb
            for m, n in matches:
                if m.distance < .75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    points.append((kp1, kp2))
        Rt = None
        if len(points) > 0:
            points = np.array(points)
            points[:, 0, :] = self.normalize(points[:, 0, :])
            points[:, 1, :] = self.normalize(points[:, 1, :])

            model, inliers = ransac((points[:, 0], points[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8, residual_threshold=0.01,
                                    max_trials=200)
            points = points[inliers]

            Rt = extract_pose(model.params)


        self.last = {'kps': kps, 'des': des}
        return points, Rt
