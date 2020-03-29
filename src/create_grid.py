"""
    File: create_grid.py
    Author: David Solanas Sanz
    TFG
"""

import os
import numpy as np
import random
import nibabel as nib
import csv
import subprocess


def create_grid(src_image):
    x, y, z, _ = src_image.shape
    slices = []
    for i in range(32, 64, 2):
        slices.append(np.rot90(src_image.dataobj[:, :, i]))
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
            dx = row[59]
            month = 0 if (row[2] == 'bl' or row[2] == '') else int(row[2][1:])
            if key in dictionary:
                # check if month is greater and update diagnosis
                val = dictionary[key]
                if val[0] < month or (val[1] == '' and dx != ''):
                    if dx != '' and dx != val[1]:
                        dictionary[key] = [month, dx]
                    else:
                        dictionary[key] = [month, val[1]]
            else:
                dictionary[key] = [month, dx]

    return dictionary


base = '/Users/david/TFG/ADNI-NN'
images = os.listdir(base)

d = create_dictionary('/Users/david/TFG/ADNIMERGE.csv')

random.shuffle(images)

train_size = len(images) * 0.9

train_dir = '/Users/david/TFG/brain_data/train'
validation_dir = '/Users/david/TFG/brain_data/validation'

count = 0

for image in images:
    path = os.path.join(base, image)
    img = nib.load(path)
    slices = create_grid(img)
    matrix = create_data_matrix(slices)
    matrix = matrix / matrix.max()

    a = image.split('_')
    patient_id = a[1] + '_' + a[2] + '_' + a[3]
    img_name = image.split(".nii.gz")[0]
    if patient_id in d:
        _, dx = d[patient_id]
        if count < train_size:
            # Save to train directories
            if dx == 'CN':
                cn_train_dir = os.path.join(train_dir, 'CN')
                file = os.path.join(cn_train_dir, img_name + '.npy')
                subprocess.call(["touch", file])
                np.save(file, matrix)

            if dx == 'Dementia':
                ad_train_dir = os.path.join(train_dir, 'AD')
                file = os.path.join(ad_train_dir, img_name + '.npy')
                subprocess.call(["touch", file])
                np.save(file, matrix)

            if dx == 'MCI':
                mci_train_dir = os.path.join(train_dir, 'MCI')
                file = os.path.join(mci_train_dir, img_name + '.npy')
                subprocess.call(["touch", file])
                np.save(file, matrix)
        else:
            # Save to validation directories
            if dx == 'CN':
                cn_val_dir = os.path.join(validation_dir, 'CN')
                file = os.path.join(cn_val_dir, img_name + '.npy')
                subprocess.call(["touch", file])
                np.save(file, matrix)
            if dx == 'Dementia':
                ad_val_dir = os.path.join(validation_dir, 'AD')
                file = os.path.join(ad_val_dir, img_name + '.npy')
                subprocess.call(["touch", file])
                np.save(file, matrix)
            if dx == 'MCI':
                mci_val_dir = os.path.join(validation_dir, 'MCI')
                file = os.path.join(mci_val_dir, img_name + '.npy')
                subprocess.call(["touch", file])
                np.save(file, matrix)

        count += 1

print(count)
