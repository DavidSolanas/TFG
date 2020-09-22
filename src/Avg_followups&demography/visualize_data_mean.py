"""
    File: visualize_data_mean.py
    Author: David Solanas Sanz
    TFG
"""

import os
import numpy as np
import keras
import matplotlib.pyplot as plt

base_dir = 'D:\\TFG\\brain_data_final_test_uniform\\train\\AD'
base_dir2 = 'D:\\TFG\\brain_data_final_test_uniform\\train\\CN'
base_dir3 = 'D:\\TFG\\brain_data_final_test_uniform\\train\\MCI'

images = os.listdir(base_dir)
images2 = os.listdir(base_dir2)
images3 = os.listdir(base_dir3)

map = np.zeros((512, 512, 3))

for file in images:
    img = keras.preprocessing.image.load_img(os.path.join(base_dir, file))
    arr = keras.preprocessing.image.img_to_array(img)
    map += arr

for file in images2:
    img = keras.preprocessing.image.load_img(os.path.join(base_dir2, file))
    arr = keras.preprocessing.image.img_to_array(img)
    map += arr

for file in images3:
    img = keras.preprocessing.image.load_img(os.path.join(base_dir3, file))
    arr = keras.preprocessing.image.img_to_array(img)
    map += arr

map /= len(images) + len(images2) + len(images3)
map /= map.max()
plt.imshow(map, cmap='gray')
plt.show()
