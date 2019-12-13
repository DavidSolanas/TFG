"""
    File: otsu_threshold.py
    Author: David Solanas Sanz
    TFG
"""
import math

import numpy as np


def threshold(src_data, dst_img):
    x, y, z, _ = src_data.shape
    top = int(src_data.max())
    total = x * y * z
    # Calculate histogram
    histogram = np.empty(top + 1, int)
    values = np.array(range(0, top + 1))
    for i in range(0, x):
        for j in range(0, y):
            for k in range(0, z):
                val = int(src_data[i, j, k][0])
                histogram[val] += 1

    best_threshold = 0
    sumb = .0
    wb = .0
    max_inter_var = 0.0
    sum1 = np.dot(values, histogram)
    for _threshold in values:
        wf = total - wb
        if wb > 0 and wf > 0:
            mf = (sum1 - sumb) / wf
            val = wb * wf * ((sumb / wb) - mf) * ((sumb / wb) - mf)
            if val >= max_inter_var:
                max_inter_var = val
                best_threshold = _threshold

        wb += histogram[_threshold]
        sumb += _threshold * histogram[_threshold]
    print(best_threshold)

    for i in range(0, x):
        for j in range(0, y):
            for k in range(0, z):
                dst_img.dataobj[i, j, k] = 0.0 if dst_img.dataobj[i, j, k] < best_threshold else dst_img.dataobj[
                    i, j, k]
