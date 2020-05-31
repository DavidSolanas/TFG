"""
    File: resample.py
    Author: David Solanas Sanz
    TFG
"""

import os
import sys
import nibabel as nib
import numpy as np
from nilearn import image
from skimage.transform import resize
from scipy import ndimage
import math
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation


def resample_img(source_image, target_shape, voxel_dims=[2., 2., 2.]):
    """
        This function resamples an input image to a specific (mm)
        isotropic voxels and crops it to a new dimensional pixel-grid.
        Parameters
        ----------
        source_image : nibabel.nifti1.Nifti1Image
            Nifti image to resample
        target_shape: list of float
            3 numbers to specify the dimensions of the resampled image.
        voxel_dims : list
            Length in mm for x,y, and z dimensions of each voxel.
        Returns
        -------
        resampled_img : nibabel.nifti1.Nifti1Image
            The resampled image.
        """
    voxel_transform = image.resample_img(source_image, target_affine=np.diag(voxel_dims))
    ref = np.array(voxel_transform.dataobj)
    ref = np.squeeze(ref, axis=3)
    resampled_img = resize(ref, target_shape)
    return resampled_img


def copy_data(src_dataobj, src_thresholded, dst_img):
    x, y, z = src_dataobj.shape
    # Filter the image, if a value is less than the threshold is set to 0
    for i in range(0, x):
        for j in range(0, y):
            for k in range(0, z):
                dst_img[i, j, k] = src_dataobj[i, j, k] if src_thresholded[i, j, k] == 1 else 0


def get_images(path):
    directories = os.listdir(path)
    images = []
    end = False
    for _d in directories:
        _path = os.path.join(path, _d)
        if os.path.isfile(_path):
            if _d != '.DS_Store':
                images.append(_path)
                end = True
        else:
            sub_images = get_images(_path)
            images += sub_images
        if end:
            break

    return images


####################################################
print(sys.argv)

base = sys.argv[1]
dest = sys.argv[2]
src_data = get_images(base)
print(len(src_data))
for data_path in src_data:
    img = nib.load(data_path)
    dim = (100, 100, 90)
    voxels = [2., 2., 2.]
    img_resampled = resample_img(img, dim, voxels)

    # Otsu thresholding to remove noise
    if img_resampled.max() < 1.:
        thresh = 0.015
    else:
        thresh = threshold_otsu(img_resampled, nbins=math.ceil(img_resampled.max()))
    thresholded = img_resampled > thresh

    connectivity = np.ones(shape=(3, 3))
    # Fill holes in the brain image
    for z in range(0, 90):
        img_fill_holes = ndimage.binary_fill_holes(thresholded[:, :, z]).astype(int)
        eroded = erosion(img_fill_holes, connectivity)
        dilated = dilation(eroded, connectivity)
        for x in range(0, 100):
            for y in range(0, 100):
                thresholded[x, y, z] = dilated[x, y]

    copy_data(img_resampled, thresholded, img_resampled)

    # Save image
    name_img = data_path.split('\\')[-1].split('.')[0] + '.npy'
    print(name_img)
    np.save(os.path.join(dest, name_img), img_resampled)
