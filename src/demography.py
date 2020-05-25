"""
    File: demography.py
    Author: David Solanas Sanz
    TFG
"""

import os
import csv
import numpy as np


def create_dictionary(filename):
    dictionary = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            key = row[1]
            dx = row[59]
            age = float(row[8]) if row[8] != '' else -1
            gender = row[9]
            month = 0 if (row[2] == 'bl' or row[2] == '') else int(row[2][1:])
            if key in dictionary:
                # check if month is greater and update diagnosis
                val = dictionary[key]
                if val[0] < month or (val[1] == '' and dx != ''):
                    if dx != '' and dx != val[1]:
                        dictionary[key] = [month, dx, age, gender]
                    else:
                        dictionary[key] = [month, val[1], age, gender]
            else:
                dictionary[key] = [month, dx, age, gender]

    return dictionary


patients = os.listdir('D:\\TFG\\ADNI')
imaging_studies = os.listdir('D:\\TFG\\ADNI-NN')
d = create_dictionary('D:\\TFG\\ADNIMERGE.csv')

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
    len(female_ad_ages), np.mean(female_ad_ages), np.std(female_ad_ages), female_ad_ages.min(), female_ad_ages.max()))
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
    len(female_cn_ages), np.mean(female_cn_ages), np.std(female_cn_ages), female_cn_ages.min(), female_cn_ages.max()))
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

n_studies = t_ad + t_mci + t_cn + v_ad + v_mci + v_cn
ad_studies = t_ad + v_ad
cn_studies = t_cn + v_cn
mci_studies = t_mci + v_mci

print('Number of imaging studies: %d' % n_studies)
print('Number of AD imaging studies: %d' % ad_studies)
print('Number of MCI imaging studies: %d' % mci_studies)
print('Number of CN imaging studies: %d' % cn_studies)
