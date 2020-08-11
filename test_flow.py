import numpy as np

from flow import calc_avg_flow


def test_average_simple():
    flow = np.zeros((5, 5, 3))

    avg = calc_avg_flow(flow, row_stride=1, col_stride=1)

    assert np.all(avg == 0)

    for i in range(0, 5):
        for j in range(0, 5):
            flow[i, j] = 1

    avg = calc_avg_flow(flow, row_stride=1, col_stride=1)

    assert np.all(avg == 1)
    assert avg.shape == (5, 5, 3)


def test_average_flow():
    flow = np.zeros((25, 25, 3))

    for i in range(0, 25, 5):
        for j in range(0, 25, 5):
            flow[i, j] = 1

    avg = calc_avg_flow(flow, row_stride=5, col_stride=5)

    assert np.all(avg == .04)
    assert avg.shape == (5, 5, 3)

    flow2 = np.zeros((20, 40, 3))

    for i in range(0, 20, 5):
        for j in range(0, 40, 5):
            flow2[i, j] = 1

    avg = calc_avg_flow(flow2, row_stride=5, col_stride=5)

    assert avg.shape == (4, 8, 3)
    assert np.all(avg == .04)
