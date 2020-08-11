import cv2 as cv
import numpy as np
import glob
import re

from tqdm import tqdm
from multiprocessing.pool import ThreadPool


def calc_flow(img1, img2):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3,
                                       5, 1.2, 0)

    return flow


def calc_avg_flow(flow, row_stride=32, col_stride=32):
    h, w = flow.shape[:2]
    if h % row_stride != 0 or w % col_stride != 0:
        raise ValueError("Strids not perfectly divisible")
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
    y, x = np.mgrid[h_step//2:h:h_step, w_step//2:w:w_step].reshape(2, -1)
    ys, xs = np.mgrid[0:h_size, 0:w_size].reshape(2, -1).astype(int)
    fx, fy = flow[ys, xs].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv.polylines(img, lines, 0, (0, 0, 255), thickness=2)
    for (x1, y1), (x2, y2) in lines:
        cv.circle(img, (x1, y1), 2, (0, 0, 255), -1)
    return img


def natural_sort(names):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(names, key=alphanum_key)


def generate_data(folder, y_data, row_stride=32, col_stride=32, save_name='flows'):
    files = natural_sort(glob.glob(folder))

    assert len(files) > 0, "Folder empty, make sure to provide a correct path"

    print(f'loading {len(files)} files')
    flows = []
    for file1, file2 in tqdm(zip(files, files[1:]), total=len(files)-1):
        frame1 = cv.imread(file1)
        frame2 = cv.imread(file2)

        flow = calc_flow(frame1, frame2)
        avg_flow = calc_avg_flow(flow, row_stride=row_stride, col_stride=col_stride)
        flows.append(avg_flow)

    y_data = np.loadtxt(y_data)
    mean_speeds = []
    for s1, s2 in zip(y_data, y_data[1:]):
        mean_speeds.append(np.mean([s1, s2]))

    flows = np.array(flows)
    mean_speeds = np.array(mean_speeds)

    assert len(mean_speeds) == len(flows), f'X len {len(flows)}, y len {len(mean_speeds)}'

    np.save(f'{save_name}_r{row_stride}_c{col_stride}.npy', flows)
    np.save(f'flows_r{row_stride}_c{col_stride}_speed.npy', mean_speeds)
    flows = flows.reshape(flows.shape[0], -1)
    return flows, mean_speeds


def threaded_flow(args):
    file1, file2, row_stride, col_stride = args[0], args[1], args[2], args[3]
    frame1 = cv.imread(file1)
    frame2 = cv.imread(file2)

    flow = calc_flow(frame1, frame2)
    avg_flow = calc_avg_flow(flow, row_stride=row_stride, col_stride=col_stride)

    return avg_flow


def generate_data_threaded(folder, y_data, row_stride=32, col_stride=32, save_name='flows'):
    files = natural_sort(glob.glob(folder))

    assert len(files) > 0, "Folder empty, make sure to provide a correct path"

    print(f'loading {len(files)} files')
    map_length = len(files[1:])
    mapped = list(zip(files, files[1:], [row_stride] * map_length, [col_stride] * map_length))
    with ThreadPool(processes=8) as p:
       result = list(tqdm(p.imap(threaded_flow, mapped), total=len(mapped)))
       return result


if __name__ == '__main__':
    cap = cv.videocapture('video/train.mp4')
    ret, frame1 = cap.read()

    flows = []
    while cap.isopened():
        ret, frame2 = cap.read()
        if not ret:
            break

        flow = calc_flow(frame1, frame2)
        im = draw_flow(frame1, flow)
        avg_flow = calc_avg_flow(flow)
        im = draw_avg_flow(frame1, avg_flow)
        flows.append(avg_flow)
        cv.imshow('avg flow', im)
        cv.waitkey(30)

        frame1 = frame2

    np.save('flows.npy', np.array(flows))
    cv.destroyallwindows()
