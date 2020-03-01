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

    # Restore correct values in header
    voxel_transform.header['dim'][5:] = 0
    voxel_transform.header['pixdim'][5:] = 0
    voxel_transform.header['xyzt_units'] = 2

    img_shape = np.array(voxel_transform.header['dim'][1:4])

    target_affine = voxel_transform.affine.copy()
    center_x = img_shape[0] // 2 - 50
    center_y = img_shape[1] // 2 - 50
    center_z = img_shape[2] // 2 - 45
    target_affine[0, 3] = center_x * target_affine[0, 0] + target_affine[0, 3]
    target_affine[1, 3] = center_y * target_affine[1, 1] + target_affine[1, 3]
    target_affine[2, 3] = center_z * target_affine[2, 2] + target_affine[2, 3]

    new_img = image.resample_img(voxel_transform, target_affine=target_affine, target_shape=target_shape)

    return new_img


def copy_data(src_dataobj, src_thresholded, dst_img):
    x, y, z, _ = src_dataobj.shape
    # Filter the image, if a value is less than the threshold is set to 0
    for i in range(0, x):
        for j in range(0, y):
            for k in range(0, z):
                dst_img.dataobj[i, j, k] = src_dataobj[i, j, k] if src_thresholded[i, j, k] == 1 else 0


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


####################################################
print(sys.argv)

base = sys.argv[1]
dest = sys.argv[2]
src_data = get_images(base)

for data_path in src_data:
    img = nib.load(data_path)
    print(data_path)
    dim = (100, 100, 90)
    voxels = [2., 2., 2.]
    img_resampled = resample_img(img, dim, voxels)

    # Otsu thresholding to remove noise
    data = img_resampled.dataobj
    if data.max() < 1.:
        thresh = 0.015
    else:
        thresh = threshold_otsu(data, nbins=math.ceil(data.max()))
    thresholded = img_resampled.dataobj > thresh

    connectivity = np.ones(shape=(3, 3))
    # Fill holes in the brain image
    for z in range(0, 90):
        img_fill_holes = ndimage.binary_fill_holes(np.squeeze(thresholded[:, :, z], axis=2)).astype(int)
        eroded = erosion(img_fill_holes, connectivity)
        dilated = dilation(eroded, connectivity)
        for x in range(0, 100):
            for y in range(0, 100):
                thresholded[x, y, z] = dilated[x, y]

    copy_data(img_resampled.dataobj, thresholded, img_resampled)

    # Save image
    name_img = data_path.split('/')[-1] + '.gz'
    nib.save(img_resampled, os.path.join(dest, name_img))
