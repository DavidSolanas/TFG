"""
    File: train_network.py
    Author: David Solanas Sanz
    TFG
"""

import argparse
import os
import pickle
import time

import keras
import keras.applications
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import layers
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

input_size = 512
batch_size = 8


def get_model():
    """
    Creates a deep-learning model
    Returns
    -------
    Compiled model
    """
    inception_model = keras.applications.InceptionResNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
    )

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


def get_model_memory_usage(batch_size, model):
    """
    Gets the memmory usage by the model
    Parameters
    ----------
    batch_size: Batch size used to train
    model: deep-learning model

    Returns
    -------
    Memmory usage (GB)
    """

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


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
    mem = get_model_memory_usage(batch_size, model)
    print('Memory usage: ', mem)

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
    startTime = time.time()

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

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

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
