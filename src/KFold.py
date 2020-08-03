from sklearn.model_selection import KFold
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils import to_categorical
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from keras import layers, Model
from keras.optimizers import Adam
import numpy as np
import os
import math
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from keras.callbacks import LearningRateScheduler


# Reset Keras Session
def reset_keras(model):
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


num_folds = 10

acc_per_fold = []
loss_per_fold = []


def load_patients(base_dir):
    ad_dir = os.path.join(base_dir, 'AD')  # Directory with our training cat pictures
    cn_dir = os.path.join(base_dir, 'CN')  # Directory with our training cat pictures
    mci_dir = os.path.join(base_dir, 'MCI')  # Directory with our training dog pictures

    ad_fnames = os.listdir(ad_dir)
    cn_fnames = os.listdir(cn_dir)
    mci_fnames = os.listdir(mci_dir)

    p = dict()

    paths = []

    for img_AD in ad_fnames:
        patient_id = img_AD.split('_')[1] + '_' + img_AD.split('_')[2] + '_' + img_AD.split('_')[3]
        diagnostic = 0
        p[patient_id] = diagnostic
        paths.append(os.path.join(ad_dir, img_AD))

    for img_CN in cn_fnames:
        patient_id = img_CN.split('_')[1] + '_' + img_CN.split('_')[2] + '_' + img_CN.split('_')[3]
        diagnostic = 1
        p[patient_id] = diagnostic
        paths.append(os.path.join(cn_dir, img_CN))

    for img_MCI in mci_fnames:
        patient_id = img_MCI.split('_')[1] + '_' + img_MCI.split('_')[2] + '_' + img_MCI.split('_')[3]
        diagnostic = 2
        p[patient_id] = diagnostic
        paths.append(os.path.join(mci_dir, img_MCI))

    return p, paths


def load_data(base_dir):
    ad_dir = os.path.join(base_dir, 'AD')  # Directory with our training cat pictures
    cn_dir = os.path.join(base_dir, 'CN')  # Directory with our training cat pictures
    mci_dir = os.path.join(base_dir, 'MCI')  # Directory with our training dog pictures

    ad_fnames = os.listdir(ad_dir)
    cn_fnames = os.listdir(cn_dir)
    mci_fnames = os.listdir(mci_dir)

    x = []
    y = []

    # Train dataset
    for img_AD in ad_fnames:
        _x = load_img(os.path.join(ad_dir, img_AD), target_size=(299, 299), interpolation='bicubic')
        _x = img_to_array(_x)
        x.append(_x)
        y.append(0)

    for img_CN in cn_fnames:
        _x = load_img(os.path.join(cn_dir, img_CN), target_size=(299, 299), interpolation='bicubic')
        _x = img_to_array(_x)
        x.append(_x)
        y.append(1)

    for img_MCI in mci_fnames:
        _x = load_img(os.path.join(mci_dir, img_MCI), target_size=(299, 299), interpolation='bicubic')
        _x = img_to_array(_x)
        x.append(_x)
        y.append(2)

    # Convert data to numpy array
    x = np.array(x)
    y = np.array(y)
    y = to_categorical(y, num_classes=3)

    return x, y


def get_model():
    inception_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3))

    # inception_model.load_weights('../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    for layer in inception_model.layers:
        layer.trainable = True

    x = inception_model.output
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.6)(x)
    # x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    predictions = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    print(model.summary())

    model.compile(optimizer=Adam(lr=0.0001, epsilon=0.1),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def get_model2():
    inception_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3))

    # inception_model.load_weights('../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    for layer in inception_model.layers:
        layer.trainable = True

    x = inception_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.6)(x)
    # x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    model.compile(optimizer=Adam(lr=0.0001, epsilon=0.1),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def get_model3():
    inception_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3))

    # inception_model.load_weights('../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    for layer in inception_model.layers:
        layer.trainable = True

    x = inception_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.6)(x)
    # x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    model.compile(optimizer=Adam(lr=0.0001, epsilon=0.1),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def get_model4():
    inception_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3))

    # inception_model.load_weights('../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    for layer in inception_model.layers:
        layer.trainable = True

    x = inception_model.output
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    predictions = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    print(model.summary())

    model.compile(optimizer=Adam(lr=0.0001, epsilon=0.1),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def get_model5():
    inception_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3))

    # inception_model.load_weights('../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    for layer in inception_model.layers:
        layer.trainable = True

    x = inception_model.output
    x = layers.Dropout(0.6)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    model.compile(optimizer=Adam(lr=0.0001, epsilon=0.1),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


# BEST MODEL
def get_model6():
    inception_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3))

    # inception_model.load_weights('../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    for layer in inception_model.layers:
        layer.trainable = True

    x = inception_model.output
    x = layers.Dropout(0.6)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    model.compile(optimizer=Adam(lr=0.0001, epsilon=1.),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def get_model7():
    inception_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3))

    # inception_model.load_weights('../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    for layer in inception_model.layers:
        layer.trainable = True

    x = inception_model.output
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.6)(x)
    # x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    predictions = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    print(model.summary())

    model.compile(optimizer=Adam(lr=0.0001, epsilon=0.1),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def get_model8():
    inception_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3))

    # inception_model.load_weights('../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    for layer in inception_model.layers:
        layer.trainable = True

    x = inception_model.output
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    predictions = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)

    print(model.summary())

    model.compile(optimizer=Adam(lr=0.0001, epsilon=0.1),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def load_images(patients, train_indices, test_indices, dictionary, paths):
    train_paths = []
    test_paths = []

    for train_patient in patients[train_indices]:
        aux = [x for x in paths if train_patient in x]
        train_paths = train_paths + aux

    for test_patient in patients[test_indices]:
        aux = [x for x in paths if test_patient in x]
        test_paths = test_paths + aux

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for path in train_paths:
        _x = load_img(path, target_size=(299, 299), interpolation='bicubic')
        _x = img_to_array(_x)
        x_train.append(_x)
        patient_id = path.split('/')[-1][5:15]
        y_train.append(dictionary[patient_id])

    for path in test_paths:
        _x = load_img(path, target_size=(299, 299), interpolation='bicubic')
        _x = img_to_array(_x)
        x_test.append(_x)
        patient_id = path.split('/')[-1][5:15]
        y_test.append(dictionary[patient_id])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = to_categorical(np.array(y_train), num_classes=3)
    y_test = to_categorical(np.array(y_test), num_classes=3)

    return x_train, y_train, x_test, y_test


# This function keeps the learning rate at 0.0001 for the first ten epochs
# and decreases it exponentially after that.
def exp_scheduler(epoch):
    if epoch < 10:
        return 0.0001
    else:
        return 0.0001 * math.exp(0.1 * (10 - epoch))


# This function keeps the learning rate at 0.0001 for the first ten epochs
# and decreases it exponentially after that.
def step_based_scheduler(epoch):
    # compute the learning rate for the current epoch
    exp = np.floor((1 + epoch) / 10)
    lr = .0001 * (.25 ** exp)
    # return the learning rate
    return float(lr)


def linear_scheduler(epoch):
    # compute the new learning rate based on polynomial decay
    decay = 1 - (epoch / float(20))
    lr = .0001 * decay
    # return the new learning rate
    return float(lr)


def polynomial_scheduler(epoch):
    # compute the new learning rate based on polynomial decay
    # compute the new learning rate based on polynomial decay
    decay = (1 - (epoch / float(20))) ** 2
    lr = .0001 * decay
    # return the new learning rate
    return float(lr)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only see the gpu 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
dir = '/SauronExt4/davidpet/brain_data_raw/test'
p, img_paths = load_patients(dir)
patient_list = []
for key in p:
    patient_list.append(key)

# x, y = load_data(dir)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=8,
    shear_range=np.pi / 16,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.08,
    horizontal_flip=False,
    vertical_flip=False,
    preprocessing_function=preprocess_input
)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
for train, test in kfold.split(patient_list):
    x_train, y_train, x_test, y_test = load_images(np.array(patient_list), train, test, p, img_paths)

    train_generator = train_datagen.flow(x_train, y_train, batch_size=8)
    test_generator = validation_datagen.flow(x_test, y_test, batch_size=8)

    model = get_model6()
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    callback = LearningRateScheduler(exp_scheduler)
    # Fit data to model
    history = model.fit(x_train, y_train,
                        batch_size=8,
                        epochs=20,
                        verbose=1,
                        class_weight='auto',
                        callbacks=[callback])

    # Generate generalization metrics
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1
    reset_keras(model)

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
