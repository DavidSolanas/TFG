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
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt


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
        _x, _y = np.load(os.path.join(train_ad_dir, img_AD)), 0
        _x = np.stack((_x,) * 3, axis=-1)
        x_train.append(_x)
        y_train.append(_y)
        if count > 331:
            break
        count += 1
    count = 0

    print('Loaded %.2f%% of the images...' % load)
    for img_CN in train_cn_fnames:
        _x, _y = np.load(os.path.join(train_cn_dir, img_CN)), 1
        _x = np.stack((_x,) * 3, axis=-1)
        x_train.append(_x)
        y_train.append(_y)
        if count > 331:
            break
        count += 1
    count = 0

    print('Loaded %.2f%% of the images...' % (load * 2))
    for img_MCI in train_mci_fnames:
        _x, _y = np.load(os.path.join(train_mci_dir, img_MCI)), 2
        _x = np.stack((_x,) * 3, axis=-1)
        x_train.append(_x)
        y_train.append(_y)
        if count > 331:
            break
        count += 1
    count = 0

    print('Loaded %.2f%% of the images...' % (load * 3))
    # Test dataset
    for img_AD in validation_ad_fnames:
        _x, _y = np.load(os.path.join(validation_ad_dir, img_AD)), 0
        _x = np.stack((_x,) * 3, axis=-1)
        x_test.append(_x)
        y_test.append(_y)
        if count > 31:
            break
        count += 1
    count = 0

    print('Loaded %.2f%% of the images...' % (load * 4))
    for img_CN in validation_cn_fnames:
        _x, _y = np.load(os.path.join(validation_cn_dir, img_CN)), 1
        _x = np.stack((_x,) * 3, axis=-1)
        x_test.append(_x)
        y_test.append(_y)
        if count > 31:
            break
        count += 1
    count = 0

    print('Loaded %.2f%% of the images...' % (load * 5))
    for img_MCI in validation_mci_fnames:
        _x, _y = np.load(os.path.join(validation_mci_dir, img_MCI)), 2
        _x = np.stack((_x,) * 3, axis=-1)
        x_test.append(_x)
        y_test.append(_y)
        if count > 31:
            break
        count += 1
    count = 0

    print('Load completed.')

    # Convert data to numpy array
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # Binarize y data
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    split = int(len(y_train) * .9)
    permutation_index = np.random.permutation(len(y_train))
    train_indices, validation_indices = permutation_index[:split], permutation_index[split:]

    x_train, x_validation = x_train[train_indices, :], x_train[validation_indices, :]
    y_train, y_validation = y_train[train_indices, :], y_train[validation_indices, :]

    # Reshape data to fit into the network
    # x_train = x_train.reshape(len(y_train), 512, 512, 1)
    # x_validation = x_validation.reshape(len(y_validation), 512, 512, 1)
    # x_test = x_test.reshape(len(y_test), 512, 512, 1)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def get_model():
    pre_trained_model = InceptionV3(include_top=False,
                                    weights='imagenet',
                                    input_shape=(512, 512, 3))

    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Get the last layer of Inception V3
    last_layer = pre_trained_model.get_layer('mixed10')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    # Add the top of the network
    x = layers.AveragePooling2D()(last_output)
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(x)
    # Add a dropout layer as means of regularization
    x = layers.Dropout(0.6)(x)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu')(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(3, activation='softmax')(x)

    # Create and compile the model
    model = Model(pre_trained_model.input, x)

    print(model.summary())

    model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


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
    train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.08)

    validation_datagen = ImageDataGenerator()

    # Set the batch size and calculate the number of steps per epoch
    batch_size = 8
    steps_per_epoch = len(y_train) // batch_size
    validation_steps = len(y_validation) // batch_size
    print('steps_per_epoch: ', steps_per_epoch)
    print('validation_steps: ', validation_steps)

    # Flow training images in batches of 8 using train_datagen generator
    train_generator = train_datagen.flow(x_train, y_train, shuffle=True, batch_size=batch_size)

    validation_generator = validation_datagen.flow(x_validation, y_validation, shuffle=True, batch_size=batch_size)

    # Before training, set EarlyStopping when validation_loss does not decrease
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    mc = ModelCheckpoint(filepath=model_file, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    # Train our model
    history = model.fit_generator(
        train_generator,
        epochs=100,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[es, mc]
    )

    # Plot the history of the training
    plot_history(history)

    # Load the best model
    model = load_model(model_file)
    # Get the score of the model with test dataset
    _, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_accuracy, test_accuracy))


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
