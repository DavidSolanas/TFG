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
import numpy as np
import keras
from sklearn.preprocessing import label_binarize


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

    count = 0
    # Train dataset
    for img_AD in train_AD_fnames:
        _x, _y = np.load(os.path.join(train_AD_dir, img_AD)), 0
        X_train.append(_x)
        y_train.append(_y)
        if count > 100:
            break
        count += 1
    count = 0

    for img_CN in train_CN_fnames:
        _x, _y = np.load(os.path.join(train_CN_dir, img_CN)), 1
        X_train.append(_x)
        y_train.append(_y)
        if count > 100:
            break
        count += 1
    count = 0

    for img_MCI in train_MCI_fnames:
        _x, _y = np.load(os.path.join(train_MCI_dir, img_MCI)), 2
        X_train.append(_x)
        y_train.append(_y)
        if count > 100:
            break
        count += 1
    count = 0

    # Test dataset
    for img_AD in validation_AD_fnames:
        _x, _y = np.load(os.path.join(validation_AD_dir, img_AD)), 0
        X_test.append(_x)
        y_test.append(_y)
        if count > 10:
            break
        count += 1
    count = 0

    for img_CN in validation_CN_fnames:
        _x, _y = np.load(os.path.join(validation_CN_dir, img_CN)), 1
        X_test.append(_x)
        y_test.append(_y)
        if count > 10:
            break
        count += 1
    count = 0

    for img_MCI in validation_MCI_fnames:
        _x, _y = np.load(os.path.join(validation_MCI_dir, img_MCI)), 2
        X_test.append(_x)
        y_test.append(_y)
        if count > 10:
            break
        count += 1
    count = 0

    return X_train, y_train, X_test, y_test


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

    # Define Inception V3 and freeze all its layers
    img_input = Input(shape=(512, 512, 1))
    img_conc = keras.layers.Concatenate()([img_input, img_input, img_input])

    pre_trained_model = InceptionV3(include_top=False,
                                    input_shape=(512, 512, 3),
                                    weights='imagenet', input_tensor=img_conc)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Get the last layer of Inception V3
    last_layer = pre_trained_model.get_layer('mixed10')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    # Add the top of the network
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a dropout layer as means of regularization
    x = layers.Dropout(rate=0.4)(x)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu')(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(3, activation='softmax')(x)

    # Create and compile the model
    model = Model(pre_trained_model.input, x)

    model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    # Load the data
    X_train, y_train, X_test, y_test = load_data(base_dir)

    # Convert data to numpy array
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Reshape data to fit into the network
    X_train = X_train.reshape(len(y_train), 512, 512, 1)
    X_test = X_test.reshape(len(y_test), 512, 512, 1)

    # Binarize y data
    y_train = label_binarize(y_train, classes=[0, 1, 2])
    y_test = label_binarize(y_test, classes=[0, 1, 2])

    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.08)

    # Set the batch size and calculate the number of steps per epoch
    batch_size = 8
    steps_per_epoch = len(y_train) // batch_size
    print('steps_per_epoch: ', steps_per_epoch)

    # Flow training images in batches of 8 using train_datagen generator
    train_generator = train_datagen.flow(X_train, y_train, shuffle=True,
                                         batch_size=batch_size)

    # Train our model
    history = model.fit_generator(
        train_generator,
        epochs=15,
        verbose=1,
        steps_per_epoch=steps_per_epoch)

    # Get the score of the model with test dataset
    score = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    # Save model and architecture to single file
    print("Saving model in \'%s\'..." % model_file)
    model.save(model_file)
    print("Model saved.")


if __name__ == '__main__':
    """ 
    Match input image or current life video feed with the selected template
    """
    main()
