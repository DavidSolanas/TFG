"""
    File: preprocess.py
    Author: David Solanas Sanz
    TFG
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np
from nilearn import image
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation
from skimage.transform import resize


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
    x_dim, y_dim, _ = ref.shape
    ref = resize(ref, (x_dim, y_dim, 90), order=3)
    print(ref.shape)
    final_img = np.zeros(target_shape)
    for i in range(10, 110):
        for j in range(10, 110):
            final_img[i - 10, j - 10, :] = ref[i, j, :]

    return final_img


def get_images(path):
    """
    Get all image paths to use
    Parameters
    ----------
    path: str, base path where the images are

    Returns
    -------
    List with all images (file name)
    """
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_directory", default=None, help="path to the directory where the images are stored")
    ap.add_argument("-o", "--output_directory", default=None,
                    help="path to the directory where the resampled images will be stored")
    args = ap.parse_args()

    base = None
    dest = None

    if args.input_directory is not None:
        if not os.path.isdir(args.input_directory):
            print("Directory \'%s\' does not exist" % args.input_directory)
            exit(1)
        base = args.input_directory
    else:
        print("You must specify the directory where the images are stored (see help).")
        exit(1)

    if args.output_directory is not None:
        if not os.path.isdir(args.output_directory):
            print("Directory \'%s\' does not exist" % args.output_directory)
            exit(1)
        dest = args.output_directory
    else:
        print("You must specify the directory where the resampled images will be stored (see help).")
        exit(1)

    src_data = get_images(base)

    for data_path in src_data:
        img = nib.load(data_path)
        dim = (100, 100, 90)
        voxels = [2., 2., 2.]
        img_resampled = resample_img(img, dim, voxels)

        thresh = threshold_otsu(img_resampled)
        thresholded = img_resampled > thresh

        connectivity = np.ones(shape=(3, 3))
        # Fill holes in the brain image
        for z in range(0, 90):
            img_fill_holes = ndimage.binary_fill_holes(thresholded[:, :, z]).astype(int)
            eroded = erosion(img_fill_holes, connectivity)
            dilated = dilation(eroded, connectivity)
            dilated = dilation(dilated, connectivity)
            mask = ndimage.binary_fill_holes(dilated).astype(int)
            thresholded[:, :, z] = mask

        img_resampled = img_resampled * thresholded
        img_resampled = (img_resampled - img_resampled.min()) / (img_resampled.max() - img_resampled.min())
        # Save image
        date = data_path.split('\\')[4]
        date = date[:4] + date[5:7]
        name_img = data_path.split('\\')[-1].split('.')[0]
        name_img = name_img[:15] + '_' + date + '_uniform.npy'
        print(name_img)
        np.save(os.path.join(dest, name_img), img_resampled)
