"""
    File: train_network.py
    Author: David Solanas Sanz
    TFG
"""

import os
import argparse
from keras import layers
from keras import Model, Input
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image


def load_data(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_ad_dir = os.path.join(train_dir, 'AD')  # Directory with our training cat pictures
    train_cn_dir = os.path.join(train_dir, 'CN')  # Directory with our training cat pictures
    train_mci_dir = os.path.join(train_dir, 'MCI')  # Directory with our training dog pictures

    validation_ad_dir = os.path.join(validation_dir, 'AD')  # Directory with our training cat pictures
    validation_cn_dir = os.path.join(validation_dir, 'CN')  # Directory with our training cat pictures
    validation_mci_dir = os.path.join(validation_dir, 'MCI')  # Directory with our training dog pictures

    train_ad_fnames = os.listdir(train_ad_dir)
    train_cn_fnames = os.listdir(train_cn_dir)
    train_mci_fnames = os.listdir(train_mci_dir)

    validation_ad_fnames = os.listdir(validation_ad_dir)
    validation_cn_fnames = os.listdir(validation_cn_dir)
    validation_mci_fnames = os.listdir(validation_mci_dir)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    count = 0
    load = 100 / 6
    print('Loading images...')
    # Train dataset
    for img_AD in train_ad_fnames:
        _x, _y = np.array(Image.open(os.path.join(train_ad_dir, img_AD))), 0
        _x = np.stack((_x,) * 3, axis=-1)
        _x = preprocess_input(_x)
        x_train.append(_x)
        y_train.append(_y)

    print('Loaded %.2f%% of the images...' % load)
    for img_CN in train_cn_fnames:
        _x, _y = np.array(Image.open(os.path.join(train_cn_dir, img_CN))), 1
        _x = np.stack((_x,) * 3, axis=-1)
        _x = preprocess_input(_x)
        x_train.append(_x)
        y_train.append(_y)

    print('Loaded %.2f%% of the images...' % (load * 2))
    for img_MCI in train_mci_fnames:
        _x, _y = np.array(Image.open(os.path.join(train_mci_dir, img_MCI))), 2
        _x = np.stack((_x,) * 3, axis=-1)
        _x = preprocess_input(_x)
        x_train.append(_x)
        y_train.append(_y)

    print('Loaded %.2f%% of the images...' % (load * 3))
    # Test dataset
    for img_AD in validation_ad_fnames:
        _x, _y = np.array(Image.open(os.path.join(validation_ad_dir, img_AD))), 0
        _x = np.stack((_x,) * 3, axis=-1)
        _x = preprocess_input(_x)
        x_test.append(_x)
        y_test.append(_y)

    print('Loaded %.2f%% of the images...' % (load * 4))
    for img_CN in validation_cn_fnames:
        _x, _y = np.array(Image.open(os.path.join(validation_cn_dir, img_CN))), 1
        _x = np.stack((_x,) * 3, axis=-1)
        _x = preprocess_input(_x)
        x_test.append(_x)
        y_test.append(_y)

    print('Loaded %.2f%% of the images...' % (load * 5))
    for img_MCI in validation_mci_fnames:
        _x, _y = np.array(Image.open(os.path.join(validation_mci_dir, img_MCI))), 2
        _x = np.stack((_x,) * 3, axis=-1)
        _x = preprocess_input(_x)
        x_test.append(_x)
        y_test.append(_y)

    print('Load completed.')

    # Convert data to numpy array
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train, dtype=np.uint8)
    y_test = np.array(y_test, dtype=np.uint8)

    # Binarize y data
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    split = int(len(y_train) * .9)
    permutation_index = np.random.permutation(len(y_train))
    train_indices, validation_indices = permutation_index[:split], permutation_index[split:]

    x_train, x_validation = x_train[train_indices, :], x_train[validation_indices, :]
    y_train, y_validation = y_train[train_indices], y_train[validation_indices]

    # Reshape data to fit into the network
    # x_train = x_train.reshape(len(y_train), 512, 512, 1)
    # x_validation = x_validation.reshape(len(y_validation), 512, 512, 1)
    # x_test = x_test.reshape(len(y_test), 512, 512, 1)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def get_model():
    inception_model = InceptionV3(weights='imagenet',
                                  include_top=False,
                                  input_shape=(512, 512, 3))

    x = inception_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    for layer in inception_model.layers:
        layer.trainable = True

    print(model.summary())

    model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", default=None, help="path to the directory where the images are stored")
    ap.add_argument("-m", "--model", default="model.h5", help="path to the file where the model will be stored")
    args = ap.parse_args()

    base_dir = None
    model_file = args.model

    if args.directory is not None:
        if not os.path.isdir(args.directory):
            print("Directory \'%s\' does not exist" % args.directory)
            return
        base_dir = args.directory
    else:
        print("You must specify the directory where the images are stored (see help).")
        return

    # Get the model compiled
    model = get_model()

    # Load the data
    x_train, y_train, x_validation, y_validation, x_test, y_test = load_data(base_dir)

    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rotation_range=8,
        shear_range=np.pi / 16,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.08,
        horizontal_flip=False,
        vertical_flip=False,
    )

    validation_datagen = ImageDataGenerator()

    # Set the batch size and calculate the number of steps per epoch
    input_size = 512
    batch_size = 8
    steps_per_epoch = len(y_train) // batch_size
    validation_steps = len(y_validation) // batch_size
    print('steps_per_epoch: ', steps_per_epoch)
    print('validation_steps: ', validation_steps)

    # Flow training images in batches of 8 using train_datagen generator
    train_generator = train_datagen.flow(
        x_train,
        y_train,
        shuffle=True,
        batch_size=batch_size
    )

    validation_generator = validation_datagen.flow(
        x_validation,
        y_validation,
        shuffle=False,
        batch_size=batch_size
    )

    # Before training, set EarlyStopping when validation_loss does not decrease
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=75)
    mc = ModelCheckpoint(filepath=model_file, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    # Train our model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=300,
        verbose=True,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight='auto',
        callbacks=[es, mc]
    )

    # Save the history of the training in file 'training_hist'
    with open('training_hist', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Load the best model
    # model = load_model(model_file)
    # Get the score of the model with test dataset
    # _, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    # _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    # print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_accuracy, test_accuracy))


if __name__ == '__main__':
    """ 
    Match input image or current life video feed with the selected template
    """
    # GPU memory growth and just use GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only see the gpu 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    main()
