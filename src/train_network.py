"""
    File: train_network.py
    Author: David Solanas Sanz
    TFG
"""

import os
import argparse
from keras import layers
from keras import Model
from keras.optimizers import Adam, RMSprop
from keras.applications import InceptionV3
import keras.applications
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
import pickle
import keras
import matplotlib.pyplot as plt
from skimage.transform import resize

input_size = 512
batch_size = 8


def get_model():
    inception_model = keras.applications.InceptionResNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
    )

    # inception_model.load_weights('../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in inception_model.layers:
        layer.trainable = True

    x = inception_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(3)(x)
    x = layers.BatchNormalization()(x)
    predictions = layers.Activation('softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    print(model.summary())

    model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def standarize(x):
    return (x - np.mean(x)) / np.std(x)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


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

    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(
        # rotation_range=8,
        # shear_range=np.pi / 16,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.08,
        horizontal_flip=False,
        vertical_flip=False,
    )

    validation_datagen = ImageDataGenerator()

    # Set the batch size and calculate the number of steps per epoch

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        interpolation='bicubic'
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        interpolation='bicubic'
    )
    print(train_generator.next()[0][0].min(), train_generator.next()[0][0].max())
    nb_train_samples = len(train_generator.filenames)
    nb_val_samples = len(validation_generator.filenames)

    # Before training, set EarlyStopping when validation_loss does not decrease
    es = EarlyStopping(monitor='acc', baseline=.875, patience=0)
    mc = ModelCheckpoint(filepath=model_file, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # lr = LearningRateScheduler(schedule=decayed_learning_rate)

    # Train our model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=4,
        verbose=True,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight='auto',
        initial_epoch=0,
        # callbacks=[mc]  # , lr, es]
    )

    model.save(model_file)
    # Save the history of the training in file 'training_hist'
    with open('hist', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


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
