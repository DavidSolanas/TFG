import csv
import os


def create_dictionary2(filename):
    dictionary = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            key = row[1]
            dx = row[59] if row[59] != '' else row[7]
            date = row[6]
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
            dictionary[key] = [dx, row[2]]
            dictionary[key2] = [dx, row[2]]
            dictionary[key3] = [dx, row[2]]

    return dictionary


def create_dictionary(filename):
    dictionary = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            key = row[1]
            dx = row[2]
            dictionary[key] = dx

    return dictionary


def label_data(dictionary, images, d2):
    data = []
    aux = []
    for img in images:
        patientid = img[:10]
        date = img[10:]
        if patientid + date in dictionary:
            dx = dictionary[patientid + date]
            dx[0] = d2[patientid]
            data.append((img, dx))

    return data


def get_data(filename):
    list = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            patient_id = row[1]
            acq_date = row[-3]
            acq_date = acq_date.split('/')
            month = acq_date[0] if int(acq_date[0]) >= 10 else '0' + acq_date[0]
            year = acq_date[2]
            list.append(patient_id + year + month)
    return list


# 1 patientid y -3 acq date
base = 'C:\\Users\\david\\OneDrive\\Escritorio\\Uniform-ALL-MCI.csv'
images = get_data(base)
images = sorted(images)

d = create_dictionary2('C:\\Users\\david\\OneDrive\\Escritorio\\ADNIMERGE.csv')
d2 = create_dictionary(base)
data = label_data(d, images, d2)
avg_follow_up = 0
last_patient = '002_S_0295'
n_patients = 0
n_patients_ad = 0
n_patients_cn = 0
n_patients_mci = 0
n_patients_mci2 = 0
n_patients_emci = 0
n_patients_lmci = 0
avg_follow_up_ad = 0
avg_follow_up_cn = 0
avg_follow_up_mci = 0
avg_follow_up_mci2 = 0
avg_follow_up_emci = 0
avg_follow_up_lmci = 0

for i in range(len(data)):
    img = data[i][0]
    patient_id = img[:10]
    if patient_id != last_patient:
        dx = data[i - 1][1]
        if dx[1] != 'bl' and dx[1] != 'sc':
            month = int(dx[1][1:])
            avg_follow_up += month
            n_patients += 1
            last_patient = patient_id
            if dx[0] == 'AD':
                n_patients_ad += 1
                avg_follow_up_ad += month
            if dx[0] == 'CN':
                n_patients_cn += 1
                avg_follow_up_cn += month
            if dx[0] == 'MCI':
                n_patients_mci += 1
                avg_follow_up_mci += month
            if dx[0] == 'SMC':
                n_patients_mci2 += 1
                avg_follow_up_mci2 += month
            if dx[0] == 'EMCI':
                n_patients_emci += 1
                avg_follow_up_emci += month
            if dx[0] == 'LMCI':
                n_patients_lmci += 1
                avg_follow_up_lmci += month
ad = 0
cn = 0
mci = 0
mci2 = 0
emci = 0
lmci = 0
for key in d2:
    if d2[key] == 'AD':
        ad += 1
    elif d2[key] == 'CN':
        cn += 1
    elif d2[key] == 'MCI':
        mci += 1
    elif d2[key] == 'SMC':
        mci2 += 1
    elif d2[key] == 'EMCI':
        emci += 1
    elif d2[key] == 'LMCI':
        lmci += 1

ad_img = 0
cn_img = 0
mci_img = 0
mci2_img = 0
emci_img = 0
lmci_img = 0
for img, dx in data:
    if dx[0] == 'CN':
        cn_img += 1
    if dx[0] == 'MCI':
        mci_img += 1
    if dx[0] == 'AD':
        ad_img += 1
    if dx[0] == 'SMC':
        mci2_img += 1
    if dx[0] == 'EMCI':
        emci_img += 1
    if dx[0] == 'LMCI':
        lmci_img += 1

avg_follow_up /= n_patients
avg_follow_up_ad /= n_patients_ad
avg_follow_up_mci /= n_patients_mci
avg_follow_up_cn /= n_patients_cn
avg_follow_up_mci2 /= n_patients_mci2
avg_follow_up_emci /= n_patients_emci
avg_follow_up_lmci /= n_patients_lmci
print('Number of imaging studies:', len(images))
print('AD:', ad_img)
print('MCI:', mci_img)
print('CN:', cn_img)
print('SMC:', mci2_img)
print('EMCI:', emci_img)
print('LMCI:', lmci_img)
print('Number of patients:', len(d2))
print('AD:', ad)
print('MCI:', mci)
print('CN:', cn)
print('SMC:', mci2)
print('EMCI:', emci)
print('LMCI:', lmci)
print('Avg follow-up:', avg_follow_up)
print('Avg follow-up AD:', avg_follow_up_ad)
print('Avg follow-up CN:', avg_follow_up_cn)
print('Avg follow-up MCI:', avg_follow_up_mci)
print('Avg follow-up SMC:', avg_follow_up_mci2)
print('Avg follow-up EMCI:', avg_follow_up_emci)
print('Avg follow-up LMCI:', avg_follow_up_lmci)
