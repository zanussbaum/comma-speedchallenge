import cv2
import numpy as np

from extractor import Extractor

W = 1920//2
H = 1080//2

F = 270
K = np.array([[F, 0, W//2],[0, F, H//2],[0, 0, 1]])


def process_frame(img, extractor):
    matches, Rt = extractor.extract(img)
    for pt1, pt2 in matches:
        u1, v1 = extractor.denormalize(pt1)
        u2, v2 = extractor.denormalize(pt2)
        cv2.circle(img, (u1, v1), 3, (0, 255, 0))
        cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0))

    cv2.imshow('frame', img)
    cv2.waitKey(1)

    return matches, Rt


def capture(video):
    cap = cv2.VideoCapture(video)
    ext = Extractor(K)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break

        img = cv2.resize(frame, (W,H))
        matches, pose = process_frame(img, ext)
        print(f'Matches {len(matches)}')
        print(pose)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = 'video/vo.mp4'
    capture(video)
