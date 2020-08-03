"""
    File: show_img.py
    Author: David Solanas Sanz
    TFG
"""

import os
import sys
import nibabel as nib
import numpy as np
from nilearn import image
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def show_nifti(src_image):
    dims = src_image.shape
    for _img in image.iter_img(src_image):
        # _img is now an in-memory 3D img
        slice_0 = _img.dataobj[dims[0] // 2, :, :]
        slice_1 = _img.dataobj[:, dims[1] // 2, :]
        slice_2 = _img.dataobj[:, :, dims[2] // 2]
        show_slices([slice_0, slice_1, slice_2])
        plt.suptitle("Center slices for PET image")
    plt.draw()


def create_grid(src_image):
    x, y, z, _ = src_image.shape
    slices = []
    for i in range(32, 64, 2):
        slices.append(rotate(src_image.dataobj[:, :, i], -90))
    return slices


def show_grid(slices):
    fig, axes = plt.subplots(4, 4)
    cont = 0
    for i in range(0, 4, 1):
        for j in range(0, 4, 1):
            axes[i, j].imshow(np.squeeze(slices[cont], axis=2), cmap="gray", origin="lower")
            cont += 1

    plt.suptitle("Center slices for PET image")
    plt.draw()


def get_images(path):
    directories = os.listdir(path)
    images = []
    for _d in directories:
        _path = os.path.join(path, _d)
        if os.path.isfile(_path):
            if _d != '.DS_Store':
                images.append(_path)
        else:
            sub_images = get_images(_path)
            images += sub_images

    return images


base = sys.argv[1]
src_data = get_images(base)

for data_path in src_data:
    img = nib.load(data_path)
    # Show z-axis brain image
    slices = create_grid(img)
    show_grid(slices)

    show_nifti(img)
    plt.show()
