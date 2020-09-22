"""
    File: create_grid.py
    Author: David Solanas Sanz
    TFG
"""
import argparse
import csv
import os

import keras
import numpy as np
from scipy.ndimage import rotate, measurements
from skimage.transform import resize


def create_grid(src_image):
    """
    Creates 16 brain sections from center of mass
    Parameters
    ----------
    src_image: 3D image

    Returns
    -------
    16 slices of original image
    """
    slices = []
    _, _, z = measurements.center_of_mass(src_image)
    z = int(np.round(z))
    for i in range(z - 10, z + 22, 2):
        slices.append(rotate(src_image[:, :, i], -90))
    return slices


def copy_from_to(src, dst, i1, i2, j1, j2):
    """
    Copies region in src to dst
    Parameters
    ----------
    src: source image
    dst: dest image
    i1: Begining row
    i2: End row
    j1: Begining column
    j2: End column

    Returns
    -------

    """
    i0 = 0
    for id in range(i1, i2):
        j0 = 0
        for jd in range(j1, j2):
            dst[id, jd] = src[i0, j0]
            j0 += 1
        i0 += 1


def create_data_matrix(slices):
    """
    Creates a matrix with the 16 brain sections
    Parameters
    ----------
    slices: 2D images

    Returns
    -------
    512x512 matrix
    """
    matrix = np.zeros(shape=(512, 512))
    for s1 in range(0, 4):
        for s2 in range(0, 4):
            ns = 4 * s1 + s2
            i1 = 128 * s1
            i2 = 128 * (s1 + 1)
            j1 = 128 * s2
            j2 = 128 * (s2 + 1)
            data = resize(slices[ns], (128, 128), order=3)
            copy_from_to(data, matrix, i1, i2, j1, j2)

    return matrix


def create_dictionary(filename):
    """
        Creates a dict that contains, patientid as key and diagnosis + month as value
        Parameters
        ----------
        filename: str, path to csv file

        Returns
        -------
        dictionary with all patient data
        """
    dictionary = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            phase = row[0]
            key = row[3]
            if phase == 'ADNI1':
                dx = row[11]
            elif phase == 'ADNIGO' or phase == 'ADNI2':
                dx = row[10]
            else:
                dx = row[-2]
            if dx == '1' or dx == '7' or dx == '9':
                dx = 'CN'
            elif dx == '2' or dx == '4' or dx == '8':
                dx = 'MCI'
            elif dx == '3' or dx == '5' or dx == '6':
                dx = 'AD'
            date = row[7]
            date = date.split('-')[0] + date.split('-')[1]
            key = key + date
            date2 = int(date[-2:])
            date3 = date2 - 1
            date2 = (date2 + 1) % 13
            year = int(date[:4])
            year2 = year
            if date2 == 0:
                date2 = '01'
                year += 1
            elif date2 < 10:
                date2 = '0' + str(date2)
            else:
                date2 = str(date2)

            if date3 == 0:
                date3 = '12'
                year2 -= 1
            elif date3 < 10:
                date3 = '0' + str(date3)
            else:
                date3 = str(date3)
            date2 = str(year) + date2
            date3 = str(year2) + date3
            key2 = key[:10] + date2
            key3 = key[:10] + date3
            dictionary[key] = dx
            dictionary[key2] = dx
            dictionary[key3] = dx

    dictionary['051_S_1123201202'] = 'MCI'
    dictionary['051_S_1072201202'] = 'MCI'
    dictionary['041_S_4014201107'] = 'CN'
    return dictionary


def label_data(dictionary, images):
    """
    Labels the data depending on patient's diagnosis
    Parameters
    ----------
    dictionary: Dict with patient information
    images: Names of images to label

    Returns
    -------
    Labeled data
    """
    data = []
    last_patient = ''
    aux = []
    for img in images:
        patientid = img[5:15]
        if last_patient == '':
            last_patient = patientid
            aux.append(img)
            continue
        if patientid == last_patient:
            aux.append(img)
        else:
            last_date = aux[-1][16:22]
            if last_patient + last_date in dictionary:
                dx = dictionary[last_patient + last_date]
                for a in aux:
                    data.append((a, dx))

            aux = [img]
            last_patient = patientid
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_directory", default=None, help="path to the directory where the images are stored")
    ap.add_argument("-d", "--dictionary", default=None, help="path to the csv file where the patient data is stored")
    ap.add_argument("-o", "--output_directory", default=None,
                    help="path to the directory where the preprocessed images will be stored")
    args = ap.parse_args()

    base = None
    dest = None
    dict_path = None

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

    if args.dictionary is not None:
        if not os.path.isfile(args.dictionary):
            print("File \'%s\' does not exist" % args.dictionary)
            exit(1)
        dict_path = args.dictionary
    else:
        print("You must specify the csv file where the patient data is stored (see help).")
        exit(1)

    images = os.listdir(base)

    d = create_dictionary(dict_path)
    data = label_data(d, images)

    print(len(data))
    test_size = int(len(data) * 0.9)
    val_size = int(test_size * .9)

    # train_images, val_images, test_images = split_images(images, val_size, test_size)

    train_images = data[:val_size + 7]
    val_images = data[val_size + 7:test_size - 3]
    test_images = data[test_size - 3:]

    print(len(train_images), len(val_images), len(test_images), len(train_images) + len(val_images) + len(test_images))

    train_dir = os.path.join(dest, 'train')
    validation_dir = os.path.join(dest, 'validation')
    test_dir = os.path.join(dest, 'test')

    # Creates output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    os.makedirs(os.path.join(train_dir, 'AD'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'MCI'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'CN'), exist_ok=True)

    os.makedirs(os.path.join(validation_dir, 'AD'), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, 'MCI'), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, 'CN'), exist_ok=True)

    os.makedirs(os.path.join(test_dir, 'AD'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'MCI'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'CN'), exist_ok=True)

    for image, dx in train_images:
        path = os.path.join(base, image)
        img = np.load(path)
        print(img.shape)
        slices = create_grid(img)
        matrix = create_data_matrix(slices)
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())

        img_name = image.split(".npy")[0]
        # Save to train directories
        if dx == 'CN':
            cn_train_dir = os.path.join(train_dir, 'CN')
            file = os.path.join(cn_train_dir, img_name + '.tif')
            matrix = np.stack((matrix,) * 3, axis=-1)
            keras.preprocessing.image.save_img(file, matrix)
            # np.save(file, matrix)

        if dx == 'AD':
            ad_train_dir = os.path.join(train_dir, 'AD')
            file = os.path.join(ad_train_dir, img_name + '.tif')
            matrix = np.stack((matrix,) * 3, axis=-1)
            keras.preprocessing.image.save_img(file, matrix)
            # np.save(file, matrix)

        if dx == 'MCI' or dx == 'LMCI':
            mci_train_dir = os.path.join(train_dir, 'MCI')
            file = os.path.join(mci_train_dir, img_name + '.tif')
            matrix = np.stack((matrix,) * 3, axis=-1)
            keras.preprocessing.image.save_img(file, matrix)
            # np.save(file, matrix)

    print('Stored train images.')

    for image, dx in val_images:
        path = os.path.join(base, image)
        img = np.load(path)
        slices = create_grid(img)
        matrix = create_data_matrix(slices)
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())

        img_name = image.split(".npy")[0]

        # Save to train directories
        if dx == 'CN':
            cn_val_dir = os.path.join(validation_dir, 'CN')
            file = os.path.join(cn_val_dir, img_name + '.tif')
            matrix = np.stack((matrix,) * 3, axis=-1)
            keras.preprocessing.image.save_img(file, matrix)
            # np.save(file, matrix)

        if dx == 'AD':
            ad_val_dir = os.path.join(validation_dir, 'AD')
            file = os.path.join(ad_val_dir, img_name + '.tif')
            matrix = np.stack((matrix,) * 3, axis=-1)
            keras.preprocessing.image.save_img(file, matrix)
            # np.save(file, matrix)

        if dx == 'MCI' or dx == 'LMCI':
            mci_val_dir = os.path.join(validation_dir, 'MCI')
            file = os.path.join(mci_val_dir, img_name + '.tif')
            matrix = np.stack((matrix,) * 3, axis=-1)
            keras.preprocessing.image.save_img(file, matrix)
            # np.save(file, matrix)

    print('Stored validation images.')

    for image, dx in test_images:
        path = os.path.join(base, image)
        img = np.load(path)
        slices = create_grid(img)
        matrix = create_data_matrix(slices)
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())

        img_name = image.split(".npy")[0]

        # Save to train directories
        if dx == 'CN':
            cn_test_dir = os.path.join(test_dir, 'CN')
            file = os.path.join(cn_test_dir, img_name + '.tif')
            matrix = np.stack((matrix,) * 3, axis=-1)
            keras.preprocessing.image.save_img(file, matrix)
            # np.save(file, matrix)

        if dx == 'AD':
            ad_test_dir = os.path.join(test_dir, 'AD')
            file = os.path.join(ad_test_dir, img_name + '.tif')
            matrix = np.stack((matrix,) * 3, axis=-1)
            keras.preprocessing.image.save_img(file, matrix)
            # np.save(file, matrix)

        if dx == 'MCI' or dx == 'LMCI':
            mci_test_dir = os.path.join(test_dir, 'MCI')
            file = os.path.join(mci_test_dir, img_name + '.tif')
            matrix = np.stack((matrix,) * 3, axis=-1)
            keras.preprocessing.image.save_img(file, matrix)
            # np.save(file, matrix)

    print('Stored test images.')
