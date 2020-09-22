"""
    File: calc_avg_followup.py
    Author: David Solanas Sanz
    TFG
"""
import os
import numpy as np
from src.preprocessing.create_grid import create_dictionary, label_data

if __name__ == "__main__":
    base = 'D:\\ADNI-NN'
    images = os.listdir(base)

    d = create_dictionary('D:\\DXSUM_PDXCONV_ADNIALL.csv')
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
