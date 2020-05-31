"""
    File: create_grid.py
    Author: David Solanas Sanz
    TFG
"""

import os
import numpy as np
import csv
from PIL import Image


def normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = np.array(mean)
    std = np.array(std)
    return (x - mean) / std


def create_grid(src_image):
    x, y, z = src_image.shape
    slices = []
    for i in range(32, 64, 2):
        slices.append(np.rot90(src_image[:, :, i]))
    return slices


def copy_from_to(src, dst, i1, i2, j1, j2):
    i0 = 0
    for id in range(i1, i2):
        j0 = 0
        for jd in range(j1, j2):
            dst[id, jd] = src[i0, j0]
            j0 += 1
        i0 += 1


def create_data_matrix(slices):
    matrix = np.zeros(shape=(512, 512))
    for s1 in range(0, 4):
        for s2 in range(0, 4):
            ns = 4 * s1 + s2
            i1 = 128 * s1 + 14
            i2 = 128 * (s1 + 1) - 14
            j1 = 128 * s2 + 14
            j2 = 128 * (s2 + 1) - 14
            copy_from_to(slices[ns], matrix, i1, i2, j1, j2)

    return matrix


def create_dictionary(filename):
    dictionary = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            key = row[1]
            group = row[2]
            if group == 'LMCI':
                group = 'MCI'
            dictionary[key] = group

    return dictionary


base = 'D:\\ADNI-NN'
images = os.listdir(base)

d = create_dictionary('D:\\Patient_data.csv')
print(d)
# random.shuffle(images)

train_size = len(images) * 0.9

train_dir = 'D:\\TFG\\brain_data\\train'
validation_dir = 'D:\\TFG\\brain_data\\validation'

count = 0

for image in images:
    path = os.path.join(base, image)
    img = np.load(path)
    slices = create_grid(img)
    matrix = create_data_matrix(slices)
    matrix = matrix / matrix.max()
    # matrix = np.stack((matrix,) * 3, axis=-1)
    # matrix = normalize(matrix)

    a = image.split('_')
    patient_id = a[1] + '_' + a[2] + '_' + a[3]
    img_name = image.split(".npy")[0]
    if patient_id in d:
        dx = d[patient_id]
        if count < train_size:
            # Save to train directories
            if dx == 'CN':
                cn_train_dir = os.path.join(train_dir, 'CN')
                file = os.path.join(cn_train_dir, img_name + '.tif')
                img1 = Image.fromarray(matrix)
                img1.save(file)

            if dx == 'AD':
                ad_train_dir = os.path.join(train_dir, 'AD')
                file = os.path.join(ad_train_dir, img_name + '.tif')
                img1 = Image.fromarray(matrix)
                img1.save(file)

            if dx == 'MCI':
                mci_train_dir = os.path.join(train_dir, 'MCI')
                file = os.path.join(mci_train_dir, img_name + '.tif')
                img1 = Image.fromarray(matrix)
                img1.save(file)
        else:
            # Save to validation directories
            if dx == 'CN':
                cn_val_dir = os.path.join(validation_dir, 'CN')
                file = os.path.join(cn_val_dir, img_name + '.tif')
                img1 = Image.fromarray(matrix)
                img1.save(file)

            if dx == 'AD':
                ad_val_dir = os.path.join(validation_dir, 'AD')
                file = os.path.join(ad_val_dir, img_name + '.tif')
                img1 = Image.fromarray(matrix)
                img1.save(file)

            if dx == 'MCI':
                mci_val_dir = os.path.join(validation_dir, 'MCI')
                file = os.path.join(mci_val_dir, img_name + '.tif')
                img1 = Image.fromarray(matrix)
                img1.save(file)

        count += 1

print(count)
