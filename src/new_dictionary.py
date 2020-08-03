import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def create_dictionary(filename):
    dictionary = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            key = row[1]
            date = row[6]
            dx = row[59]
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


def create_dictionary2(filename):
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
                dx = 'Dementia'

            dictionary[key] = dx
    return dictionary


d = create_dictionary2('D:\\DXSUM_PDXCONV_ADNIALL.csv')
"""
images = os.listdir('D:\\ADNI-NN-VOXEL-TH')
count = 0
for img in images:
    key = img[5:15] + img.split('_')[4]
    if key in d:
        print(key, d[key])
    else:
        count += 1
        print(key)

ad_images = os.listdir('D:\\TFG\\brain_data_voxel_th_com_npy_labels\\test\\AD')
cn_images = os.listdir('D:\\TFG\\brain_data_voxel_th_com_npy_labels\\test\\CN')
mci_images = os.listdir('D:\\TFG\\brain_data_voxel_th_com_npy_labels\\test\\MCI')

for i in range(20):
    ad = np.load(os.path.join('D:\\TFG\\brain_data_voxel_th_com_npy_labels\\test\\AD', ad_images[i]))
    cn = np.load(os.path.join('D:\\TFG\\brain_data_voxel_th_com_npy_labels\\test\\CN', cn_images[i]))
    mci = np.load(os.path.join('D:\\TFG\\brain_data_voxel_th_com_npy_labels\\test\\MCI', mci_images[i]))
    fig, axis = plt.subplots(1, 3)
    axis[0].set_title('AD')
    axis[0].imshow(ad, cmap='gray')
    axis[1].set_title('CN')
    axis[1].imshow(cn, cmap='gray')
    axis[2].set_title('MCI')
    axis[2].imshow(mci, cmap='gray')
    plt.show()
"""