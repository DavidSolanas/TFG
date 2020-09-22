"""
    File: show_first_convs.py
    Author: David Solanas Sanz
    TFG
"""

import argparse
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model


def plot_conv_weights(res):
    """
    Plots the first convolutional layer weights
    Parameters
    ----------
    res: Weights to plot

    Returns
    -------

    """
    W = res
    if len(W.shape) == 4:
        W = np.squeeze(W)

        fig, axs = plt.subplots(8, 4, figsize=(8, 8))
        axs = axs.ravel()
        for i in range(32):
            x = W[:, :, :, i]
            x = (x - x.min()) / (x.max() - x.min())
            axs[i].imshow(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=None, help="path to the file where the model is stored")
    args = ap.parse_args()

    model_file = '/models/model_cat_BUENO.h5'
    # Load the model architecture and its weights
    model = load_model(model_file)
    model2 = keras.applications.InceptionV3()

    layer_names = []
    layer_weights = []
    for i, layer in enumerate(model.layers):
        if 'conv2d' in layer.name:
            print(i, layer.name)
            layer_weights.append(layer.weights)
            layer_names.append(layer.name)

    layer_namesV3 = []
    layer_weightsV3 = []
    for i, layer in enumerate(model2.layers):
        if 'conv2d' in layer.name:
            layer_weightsV3.append(layer.weights)
            layer_namesV3.append(layer.name)

    res = layer_weights[0][0].numpy() - layer_weightsV3[0][0].numpy()
    plot_conv_weights(res)

    return res


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

    res = main()
