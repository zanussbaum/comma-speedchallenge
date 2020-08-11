import cv2 as cv
import numpy as np


def calc_flow(img1, img2):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3,
                                       5, 1.2, 0)

    return flow


def calc_avg_flow(flow, row_stride=16, col_stride=16):
    h, w = flow.shape[:2]

    num_row = h // row_stride
    num_col = w // col_stride

    avg = np.zeros((num_row, num_col, flow.shape[-1]), dtype=flow.dtype)

    # what happens if not evenly divisble?
    for i in range(num_row):
        for j in range(num_col):
            row_slice = i * row_stride
            col_slice = j * col_stride

            row_end = min((i + 1) * row_stride, h)
            col_end = min((j + 1) * col_stride, w)

            window_avg = np.mean(flow[row_slice:row_end, col_slice:col_end, :],
                                 axis=(0, 1))
            avg[i, j] = window_avg

    return avg


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]

    # remember openCV does stupid inverting of h/w
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv.polylines(img, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv.circle(img, (x1, y1), 2, (0, 255, 0), -1)
    return img


def draw_avg_flow(img, flow):
    h, w = img.shape[:2]
    h_size, w_size = flow.shape[:2]
    h_step = h // h_size
    w_step = w // w_size

    # remember openCV does stupid inverting of h/w
    y, x = np.mgrid[h_step//2:h_size*h_step:h_step, w_step//2:w_size*w_step:w_step].reshape(2, -1)
    ys, xs = np.mgrid[0:h_size, 0:w_size].reshape(2, -1).astype(int)

    fx, fy = flow[ys, xs].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv.polylines(img, lines, 0, (0, 0, 255), thickness=2)
    for (x1, y1), (x2, y2) in lines:
        cv.circle(img, (x1, y1), 2, (0, 0, 255), -1)
    return img


if __name__ == '__main__':
    frame1 = cv.imread('frames/original/frame_1.jpg')
    frame2 = cv.imread('frames/original/frame_2.jpg')

    flow = calc_flow(frame1, frame2)

    im = draw_flow(frame1, flow)
    cv.imshow('frame', im)
    cv.waitKey(0)

    avg_flow = calc_avg_flow(flow)
    im = draw_avg_flow(frame1, avg_flow)
    cv.imshow('frame', im)
    cv.waitKey(0)

    cv.destroyAllWindows()


