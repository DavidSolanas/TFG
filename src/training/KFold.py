"""
    File: train_network.py
    Author: David Solanas Sanz
    TFG
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from train_network import get_model


def reset_keras(model):
    """
    Resets keras session
    Parameters
    ----------
    model: Model to clear

    Returns
    -------

    """
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model  # this is from global space - change this as you need
    except:
        pass

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


input_size = 512
batch_size = 8


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
        rotation_range=8,
        shear_range=np.pi / 16,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.08,
        horizontal_flip=False,
        vertical_flip=False,
        preprocessing_function=normalize
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
    nb_test_samples = len(validation_generator.filenames)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    batches = 0
    for x_batch, y_batch in train_generator:
        for i in range(len(y_batch)):  # Get input
            x_train.append(x_batch[i])
            y_train.append(y_batch[i])
        batches += 1
        if batches >= nb_train_samples / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

    batches = 0
    for x_batch, y_batch in validation_generator:
        for i in range(len(y_batch)):  # Get input
            x_test.append(x_batch[i])
            y_test.append(y_batch[i])
        batches += 1
        if batches >= nb_test_samples / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)
    fold_no = 1
    for train, test in kfold.split(x, y):
        model = get_model()
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        model.fit(x[train], y[train],
                  epochs=5,
                  verbose=True,
                  batch_size=batch_size)

        # Generate generalization metrics
        scores = model.evaluate(x[test], y[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
        # reset_keras(model)

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')



if __name__ == '__main__':
    """ 
    Match input image or current life video feed with the selected template
    """
    # GPU memory growth and just use GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # only see the gpu 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    main()
