"""
    File: PET_NN.py
    Author: David Solanas Sanz
    TFG
"""

import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def load_data(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_AD_dir = os.path.join(train_dir, 'AD')  # Directory with our training cat pictures
    train_CN_dir = os.path.join(train_dir, 'CN')  # Directory with our training cat pictures
    train_MCI_dir = os.path.join(train_dir, 'MCI')  # Directory with our training dog pictures

    validation_AD_dir = os.path.join(validation_dir, 'AD')  # Directory with our training cat pictures
    validation_CN_dir = os.path.join(validation_dir, 'CN')  # Directory with our training cat pictures
    validation_MCI_dir = os.path.join(validation_dir, 'MCI')  # Directory with our training dog pictures

    train_AD_fnames = os.listdir(train_AD_dir)
    train_CN_fnames = os.listdir(train_CN_dir)
    train_MCI_fnames = os.listdir(train_MCI_dir)

    validation_AD_fnames = os.listdir(validation_AD_dir)
    validation_CN_fnames = os.listdir(validation_CN_dir)
    validation_MCI_fnames = os.listdir(validation_MCI_dir)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Train dataset
    for img_AD in train_AD_fnames:
        _x, _y = np.load(os.path.join(train_AD_dir, img_AD)), 0
        X_train.append(_x)
        y_train.append(_y)

    for img_CN in train_CN_fnames:
        _x, _y = np.load(os.path.join(train_CN_dir, img_CN)), 1
        X_train.append(_x)
        y_train.append(_y)

    for img_MCI in train_MCI_fnames:
        _x, _y = np.load(os.path.join(train_MCI_dir, img_MCI)), 2
        X_train.append(_x)
        y_train.append(_y)

    # Test dataset
    for img_AD in validation_AD_fnames:
        _x, _y = np.load(os.path.join(validation_AD_dir, img_AD)), 0
        X_test.append(_x)
        y_test.append(_y)

    for img_CN in validation_CN_fnames:
        _x, _y = np.load(os.path.join(validation_CN_dir, img_CN)), 1
        X_test.append(_x)
        y_test.append(_y)

    for img_MCI in validation_MCI_fnames:
        _x, _y = np.load(os.path.join(validation_MCI_dir, img_MCI)), 2
        X_test.append(_x)
        y_test.append(_y)

    return X_train, y_train, X_test, y_test


pre_trained_model = InceptionV3(include_top=False,
                                input_shape=(512, 512, 3),
                                weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed10')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
x = layers.Dropout(rate=0.4)(x)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
# Add a final sigmoid layer for classification
x = layers.Dense(3, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=Adam(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# Define our example directories and files
base_dir = '/content/drive/My Drive/brain_data'
X_train, y_train, X_test, y_test = load_data(base_dir)

X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = np.dstack([X_train] * 3)
X_test = np.dstack([X_test] * 3)

X_train = X_train.reshape(len(y_train), 512, 512, 3)
X_test = X_test.reshape(len(y_test), 512, 512, 3)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.08)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator()

batch_size = 8
steps_per_epoch = int(len(y_train) / batch_size)
validation_steps = int(len(y_test) / batch_size)

# Flow training images in batches of 8 using train_datagen generator
train_generator = train_datagen.flow(X_train, y_train, shuffle=True,
                                     batch_size=batch_size)

# Flow validation images in batches of 8 using test_datagen generator
validation_generator = test_datagen.flow(X_test, y_test, shuffle=True,
                                         batch_size=batch_size)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps)
