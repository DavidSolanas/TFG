import csv
import os
import numpy as np


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
            dictionary[key] = [dx, row[6]]
            dictionary[key2] = [dx, row[6]]
            dictionary[key3] = [dx, row[6]]

    return dictionary


def label_data(dictionary, images):
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
                # if len(aux) > 1:
                for a in aux:
                    data.append((a, dx))

            aux = [img]
            last_patient = patientid
    return data


base = 'D:\\ADNI-NN'
images = os.listdir(base)

d = create_dictionary2('D:\\DXSUM_PDXCONV_ADNIALL.csv')
data = label_data(d, images)
print(len(data))

avg_follow_up = 0
last_patient = '002_S_0295'
n_patients = 0
ad = 0
mci = 0
cn = 0
avg_follow_up_ad = 0
avg_follow_up_cn = 0
avg_follow_up_mci = 0
follow_up = []
follow_up_ad = []
follow_up_mci = []
follow_up_cn = []
for i in range(len(data)):
    img = data[i][0]
    patient_id = img[5:15]
    if patient_id != last_patient:
        dx = data[i - 1][1]
        if dx[1] != 'bl' and dx[1] != 'sc':
            month = int(dx[1][1:])
            avg_follow_up += month
            follow_up.append(month)
            n_patients += 1
            last_patient = patient_id
            if dx[0] == 'Dementia':
                ad += 1
                avg_follow_up_ad += month
                follow_up_ad.append(month)
            if dx[0] == 'CN':
                cn += 1
                avg_follow_up_cn += month
                follow_up_cn.append(month)
            if dx[0] == 'MCI':
                mci += 1
                avg_follow_up_mci += month
                follow_up_mci.append(month)

follow_up = np.array(follow_up)
follow_up_ad = np.array(follow_up_ad)
follow_up_mci = np.array(follow_up_mci)
follow_up_cn = np.array(follow_up_cn)

avg_follow_up /= n_patients
avg_follow_up_ad /= ad
avg_follow_up_mci /= mci
avg_follow_up_cn /= cn
print('Avg follow-up:', avg_follow_up, '+-', np.std(follow_up))
print('Avg follow-up AD:', avg_follow_up_ad, '+-', np.std(follow_up_ad))
print('Avg follow-up MCI:', avg_follow_up_mci, '+-', np.std(follow_up_mci))
print('Avg follow-up CN:', avg_follow_up_cn, '+-', np.std(follow_up_cn))
print(ad, mci, cn, n_patients)
