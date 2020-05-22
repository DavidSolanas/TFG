"""
    File: plot_history.py
    Author: David Solanas Sanz
    TFG
"""

import matplotlib.pyplot as plt
import pickle


def plot_history(history):
    plt.figure()
    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


history = pickle.load(open('training_hist', "rb"))
plot_history(history)
