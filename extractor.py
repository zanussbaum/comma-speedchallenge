import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
flann_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2


def make_homogenous(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


class Extractor:
    def __init__(self, K):
        self.extractor = cv2.ORB_create(nfeatures=50000)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(K)

    def normalize(self, pts):
        # K 3 x 3, pts is n x 3
        # want output to be n x 2
        # K dot pts.T == 3 X N
        return np.dot(self.Kinv, make_homogenous(pts).T).T[:, :2]

    def denormalize(self, pts):
        denormed = np.dot(self.K, np.array([pts[0], pts[1], 1]).T)
        return int(round(denormed[0])), int(round(denormed[1])) 

    def extract(self, img):
        # changing to gray scale
        img = np.mean(img, axis=2).astype(np.uint8)
        features = cv2.goodFeaturesToTrack(img, 3000, minDistance=3,
                                           qualityLevel=.01)
        kps, des = self.extractor.detectAndCompute(img, None,
                                                   descriptors=features)
        points = []
        if self.last is not None:
            matches = self.matcher.knnMatch(des, self.last['des'], k=2)
            # lowe's ratio test OpenCV: rb.gy/w5yrmb
            for m, n in matches:
                if m.distance < .75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    points.append((kp1, kp2))
        if len(points) > 0:
            points = np.array(points)
            points[:, 0, :] = self.normalize(points[:, 0, :])
            points[:, 1, :] = self.normalize(points[:, 1, :])

            model, inliers = ransac((points[:, 0], points[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8, residual_threshold=1,
                                    max_trials=200)
            points = points[inliers]
            print(model.params)

        self.last = {'kps': kps, 'des': des}
        return points
