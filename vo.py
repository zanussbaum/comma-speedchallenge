import cv2
import numpy as np

from frame import Frame, match_frames, denormalize

W = 1920//2
H = 1080//2

F = 270
K = np.array([[F, 0, W//2],[0, F, H//2],[0, 0, 1]])

def triangulate(pose1, pose2, pts1, pts2):
    points4d = np.zeros((pts1.shape[0], 4))
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]

        _, _, vt = np.linalg.svd(A)
        points4d[i] = vt[3]

    return points4d

def display_frame(img, kp1, kp2):
    for pt1, pt2 in zip(kp1, kp2):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        cv2.circle(img, (u1, v1), 3, (0, 255, 0))
        cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0))

    cv2.imshow('frame', img)
    cv2.waitKey(1)


def capture(video):
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret is False:
            break

        img = cv2.resize(frame, (W,H))
        frame = Frame(img, K)
        frames.append(frame)

        if len(frames) < 2:
            continue


        f1 = frames[-1]
        f2 = frames[-2]
        idx1, idx2, Rt = match_frames(f1, f2, K)

        f1.pose = np.dot(Rt, f2.pose)

        points4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
        points4d /= points4d[:, 3:]

        display_frame(img, f1.kps[idx1], f2.kps[idx2])


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = 'video/vo.mp4'
    capture(video)
