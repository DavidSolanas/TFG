import argparse
import os
import datetime
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def load_data(path):
    """
    Loads some images to analyze model's prediction capacity
    Parameters
    ----------
    path: Path where the images are stored

    Returns
    -------
    Images, labels and dates of the examination
    """
    fnames = os.listdir(path)
    fnames = sorted(fnames)
    print(fnames)
    x = []
    y = []
    dates = []
    for f in fnames:
        label = f.split('.')[0].split('-')[-1]

        # {'AD': 0, 'CN': 1, 'MCI': 2}
        if label == 'AD':
            label = 0
        elif label == 'CN':
            label = 1
        else:
            label = 2

        img = load_img(os.path.join(path, f))
        img = img_to_array(img)
        x.append(img)
        y.append(label)
        date = f.split('_')[4]
        date = datetime.datetime(int(date[:4]), int(date[4:]), 1)
        dates.append(date)

    return x, y, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", default=None, help="path to the directory where the images are stored")
    ap.add_argument("-m", "--model", default=None, help="path to the file where the model is loaded")
    args = ap.parse_args()

    base_dir = None
    model_file = None

    if args.directory is not None:
        if not os.path.isdir(args.directory):
            print("Directory \'%s\' does not exist" % args.directory)
            return
        base_dir = args.directory
    else:
        print("You must specify the directory where the images are stored (see help).")
        return

    if args.model is not None:
        if not os.path.isfile(args.model):
            print("File \'%s\' does not exist" % args.model)
            return
        model_file = args.model
    else:
        print("You must specify the file where the model is loaded (see help).")
        return

    # Load data: x will contain images, y will contain labels
    x, y, dates = load_data(base_dir)

    # Load the model architecture and its weights
    model = load_model(model_file)

    # idx_of_first_ad = np.argmin(y)

    for i, input_img in enumerate(x):
        j, k, l = input_img.shape
        y_pred = model.predict(input_img.reshape(1, j, k, l))
        y_pred_class = np.argmax(y_pred)
        print('y_pred: {}, y_true: {}, date_diff: {}'.format(y_pred_class, y[i], dates[i] - dates[-1]))


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
