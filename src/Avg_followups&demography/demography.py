"""
    File: demography.py
    Author: David Solanas Sanz
    TFG
"""

import os
import numpy as np
from src.preprocessing.create_grid import create_dictionary


def get_patients_and_images():
    """
    Gets all patients and images from files
    Returns
    -------
    image and patient names
    """
    base_train_AD = 'D:/TFG/brain_data/train/AD'
    base_train_MCI = 'D:/TFG/brain_data/train/MCI'
    base_train_CN = 'D:/TFG/brain_data/train/CN'

    base_val_AD = 'D:/TFG/brain_data/validation/AD'
    base_val_MCI = 'D:/TFG/brain_data/validation/MCI'
    base_val_CN = 'D:/TFG/brain_data/validation/CN'

    base_test_AD = 'D:/TFG/brain_data/test/AD'
    base_test_MCI = 'D:/TFG/brain_data/test/MCI'
    base_test_CN = 'D:/TFG/brain_data/test/CN'

    images_train_AD = os.listdir(base_train_AD)
    images_train_MCI = os.listdir(base_train_MCI)
    images_train_CN = os.listdir(base_train_CN)

    images_validation_AD = os.listdir(base_val_AD)
    images_validation_MCI = os.listdir(base_val_MCI)
    images_validation_CN = os.listdir(base_val_CN)

    images_test_AD = os.listdir(base_test_AD)
    images_test_MCI = os.listdir(base_test_MCI)
    images_test_CN = os.listdir(base_test_CN)

    images_train = images_train_AD + images_train_MCI + images_train_CN
    images_validation = images_validation_AD + images_validation_MCI + images_validation_CN
    images_test = images_test_AD + images_test_MCI + images_test_CN

    images = images_train + images_validation + images_test
    patients = []
    for img in images:
        patientId = img[5:15]
        patients.append(patientId)

    patients = list(set(patients))
    return images, patients


if __name__ == "__main__":
    imaging_studies, patients = get_patients_and_images()
    d = create_dictionary('D:\\ADNIMERGE.csv')

    male_ages = []
    female_ages = []

    male_ad_ages = []
    male_cn_ages = []
    male_mci_ages = []

    female_ad_ages = []
    female_cn_ages = []
    female_mci_ages = []

    for patient_id in patients:
        if patient_id in d:
            if d[patient_id][2] == -1:
                continue
            if d[patient_id][3] == 'Male':
                if d[patient_id][1] == 'Dementia':
                    male_ad_ages.append(d[patient_id][2])
                if d[patient_id][1] == 'MCI':
                    male_mci_ages.append(d[patient_id][2])
                if d[patient_id][1] == 'CN':
                    male_cn_ages.append(d[patient_id][2])
                male_ages.append(d[patient_id][2])

            if d[patient_id][3] == 'Female':
                if d[patient_id][1] == 'Dementia':
                    female_ad_ages.append(d[patient_id][2])
                if d[patient_id][1] == 'MCI':
                    female_mci_ages.append(d[patient_id][2])
                if d[patient_id][1] == 'CN':
                    female_cn_ages.append(d[patient_id][2])
                female_ages.append(d[patient_id][2])

    female_ages = np.array(female_ages)
    male_ages = np.array(male_ages)
    male_cn_ages = np.array(male_cn_ages)
    male_mci_ages = np.array(male_mci_ages)
    male_ad_ages = np.array(male_ad_ages)
    female_cn_ages = np.array(female_cn_ages)
    female_mci_ages = np.array(female_mci_ages)
    female_ad_ages = np.array(female_ad_ages)

    print('Number of male AD patients: %d, Average age: %.2f +- %.2f   (%d - %d)' % (
        len(male_ad_ages), np.mean(male_ad_ages), np.std(male_ad_ages), male_ad_ages.min(), male_ad_ages.max()))
    print('Number of female AD patients: %d, Average age: %.2f +- %2.f   (%d - %d)' % (
        len(female_ad_ages), np.mean(female_ad_ages), np.std(female_ad_ages), female_ad_ages.min(),
        female_ad_ages.max()))
    print()

    print('Number of male MCI patients: %d, Average age: %.2f +- %.2f   (%d - %d)' % (
        len(male_mci_ages), np.mean(male_mci_ages), np.std(male_mci_ages), male_mci_ages.min(), male_mci_ages.max()))
    print('Number of female MCI patients: %d, Average age: %.2f +- %2.f   (%d - %d)' % (
        len(female_mci_ages), np.mean(female_mci_ages), np.std(female_mci_ages), female_mci_ages.min(),
        female_mci_ages.max()))
    print()

    print('Number of male CN patients: %d, Average age: %.2f +- %.2f   (%d - %d)' % (
        len(male_cn_ages), np.mean(male_cn_ages), np.std(male_cn_ages), male_cn_ages.min(), male_cn_ages.max()))
    print('Number of female CN patients: %d, Average age: %.2f +- %2.f   (%d - %d)' % (
        len(female_cn_ages), np.mean(female_cn_ages), np.std(female_cn_ages), female_cn_ages.min(),
        female_cn_ages.max()))
    print()

    print('Number of male patients: %d, Average age: %.2f +- %.2f   (%d - %d)' % (
        len(male_ages), np.mean(male_ages), np.std(male_ages), male_ages.min(), male_ages.max()))
    print('Number of female patients: %d, Average age: %.2f +- %2.f   (%d - %d)' % (
        len(female_ages), np.mean(female_ages), np.std(female_ages), female_ages.min(), female_ages.max()))
    print()

    print('Number of patients: %d' % (len(male_ages) + len(female_ages)))
    print()

    t_ad = len(os.listdir('D:\\TFG\\brain_data\\train\\AD'))
    t_mci = len(os.listdir('D:\\TFG\\brain_data\\train\\MCI'))
    t_cn = len(os.listdir('D:\\TFG\\brain_data\\train\\CN'))
    v_ad = len(os.listdir('D:\\TFG\\brain_data\\validation\\AD'))
    v_mci = len(os.listdir('D:\\TFG\\brain_data\\validation\\MCI'))
    v_cn = len(os.listdir('D:\\TFG\\brain_data\\validation\\CN'))
    tt_ad = len(os.listdir('D:\\TFG\\brain_data\\test\\AD'))
    tt_mci = len(os.listdir('D:\\TFG\\brain_data\\test\\MCI'))
    tt_cn = len(os.listdir('D:\\TFG\\brain_data\\test\\CN'))

    n_studies = t_ad + t_mci + t_cn + v_ad + v_mci + v_cn + tt_ad + tt_mci + tt_cn
    ad_studies = t_ad + v_ad + tt_ad
    cn_studies = t_cn + v_cn + tt_cn
    mci_studies = t_mci + v_mci + tt_mci

    print('Number of imaging studies: %d' % n_studies)
    print('Number of AD imaging studies: %d' % ad_studies)
    print('Number of MCI imaging studies: %d' % mci_studies)
    print('Number of CN imaging studies: %d' % cn_studies)
