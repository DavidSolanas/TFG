"""
    File: resample.py
    Author: David Solanas Sanz
    TFG
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import image, plotting
import nibabel as nib

from src.Otsu import otsu_threshold


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

    plotting.show()


"""
Code from: https://github.com/nipy/nibabel/issues/670
"""


def rescale_affine(input_affine, voxel_dims=[1, 1, 1], target_center_coords=None):
    """
    This function uses a generic approach to rescaling an affine to arbitrary
    voxel dimensions. It allows for affines with off-diagonal elements by
    decomposing the affine matrix into u,s,v (or rather the numpy equivalents)
    and applying the scaling to the scaling matrix (s).

    Parameters
    ----------
    input_affine : np.array of shape 4,4
        Result of nibabel.nifti1.Nifti1Image.affine
    voxel_dims : list
        Length in mm for x,y, and z dimensions of each voxel.
    target_center_coords: list of float
        3 numbers to specify the translation part of the affine if not using the same as the input_affine.

    Returns
    -------
    target_affine : 4x4matrix
        The resampled image.
    """
    # Inicializar target_affine
    target_affine = input_affine.copy()

    # Descomposición en valores singulares de la matriz M =  U * S * V
    # Descompone la matriz M en la suma de productos de vectores
    # u -> Vectores unitarios (base ortogonal), genera el espacio de filas de M.
    # s -> Vector que contiene los valores singulares, ordenados de forma decreciente
    # v -> Vectores unitarios (base ortogonal), genera el espacio de columnas de N
    u, s, v = np.linalg.svd(target_affine[:3, :3], full_matrices=False)

    # Reescalado de la imagen al tamaño (en mm) de los vóxeles especificado
    s = voxel_dims

    # Reconstruir target_affine, @ es el operador de producto matricial
    target_affine[:3, :3] = u @ np.diag(s) @ v

    # Set the translation component of the affine computed from the input
    # image affine if coordinates are specified by the user.
    if target_center_coords is not None:
        target_affine[:3, 3] = target_center_coords
    return target_affine


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

    target_affine = source_image.affine.copy()

    # Calculate the translation part of the affine
    spatial_dimensions = (source_image.header['dim'] * source_image.header['pixdim'])[1:4]

    # Calculate the translation affine as a proportion of the real world
    # spatial dimensions
    image_center_as_prop = source_image.affine[0:3, 3] / spatial_dimensions

    # Calculate the equivalent center coordinates in the target image
    dimensions_of_target_image = (np.array(voxel_dims) * np.array(target_shape))

    target_center_coords = dimensions_of_target_image * image_center_as_prop

    # Readjust center coords
    target_center_coords = source_image.affine[0:3, 3] + (source_image.affine[0:3, 3] - target_center_coords) / 2.

    # target_center_coords[1] += 5
    target_affine = rescale_affine(target_affine, voxel_dims, target_center_coords)

    resampled_img = image.resample_img(img, target_affine, target_shape=target_shape)

    # Restore correct values in header
    resampled_img.header['pixdim'] = img.header['pixdim']
    resampled_img.header['dim'][5:] = 0
    resampled_img.header['xyzt_units'] = 2

    return resampled_img


####################################################

base_dir = '/Users/david/Desktop/PET images'
data_path = os.path.join(base_dir,
                         'scanner_prueba2.nii')

img = nib.load(data_path)
dim = (100, 100, 90)
voxels = [2., 2., 2.]

resampled_img2 = resample_img(img, dim, voxels)

resampled_img3 = resample_img(img, dim, voxels)

otsu_threshold.threshold(resampled_img2.dataobj, resampled_img3)

show_nifti(img)
show_nifti(resampled_img2)
show_nifti(resampled_img3)

nib.save(resampled_img2, os.path.join(base_dir, 'prueba_resampled2.nii.gz'))
