"""
    File: show_results.py
    Author: David Solanas Sanz
    TFG
"""

import os
import argparse
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from vis.visualization import visualize_saliency
from vis.utils import utils
import keras
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


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

    load = 100 / 6
    print('Loading images...')
    # Train dataset
    for img_AD in train_ad_fnames:
        _x, _y = np.load(os.path.join(train_ad_dir, img_AD)), 0
        # _x = np.stack((_x,) * 3, axis=-1)
        x_train.append(_x)
        y_train.append(_y)

    print('Loaded %.2f%% of the images...' % load)
    for img_CN in train_cn_fnames:
        _x, _y = np.load(os.path.join(train_cn_dir, img_CN)), 1
        # _x = np.stack((_x,) * 3, axis=-1)
        x_train.append(_x)
        y_train.append(_y)

    print('Loaded %.2f%% of the images...' % (load * 2))
    for img_MCI in train_mci_fnames:
        _x, _y = np.load(os.path.join(train_mci_dir, img_MCI)), 2
        # _x = np.stack((_x,) * 3, axis=-1)
        x_train.append(_x)
        y_train.append(_y)

    print('Loaded %.2f%% of the images...' % (load * 3))
    # Test dataset
    for img_AD in validation_ad_fnames:
        _x, _y = np.load(os.path.join(validation_ad_dir, img_AD)), 0
        # _x = np.stack((_x,) * 3, axis=-1)
        x_test.append(_x)
        y_test.append(_y)

    print('Loaded %.2f%% of the images...' % (load * 4))
    for img_CN in validation_cn_fnames:
        _x, _y = np.load(os.path.join(validation_cn_dir, img_CN)), 1
        # _x = np.stack((_x,) * 3, axis=-1)
        x_test.append(_x)
        y_test.append(_y)

    print('Loaded %.2f%% of the images...' % (load * 5))
    for img_MCI in validation_mci_fnames:
        _x, _y = np.load(os.path.join(validation_mci_dir, img_MCI)), 2
        # _x = np.stack((_x,) * 3, axis=-1)
        x_test.append(_x)
        y_test.append(_y)

    print('Load completed.')

    # Convert data to numpy array
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    permutation_index = np.random.permutation(len(y_train))
    x_train, y_train = x_train[permutation_index, :], y_train[permutation_index]

    # Reshape data to fit into the network
    x_train = x_train.reshape(len(y_train), 512, 512, 1)
    x_test = x_test.reshape(len(y_test), 512, 512, 1)

    return x_train, y_train, x_test, y_test


def get_class(y):
    y_string = []
    for i in range(len(y)):
        if y[i] == 0:
            y_string.append('AD')
        if y[i] == 1:
            y_string.append('non-AD/MCI')
        if y[i] == 2:
            y_string.append('MCI')

    return y_string


def plot_roc_curve(model, x, y):
    # Plot linewidth.
    lw = 2
    n_classes = 3
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = model.predict(x)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    classes = ['AD', 'non-AD/MCI', 'MCI']
    # Plot all ROC curves
    plt.figure(1)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC for {0}: AUC = {1:0.2f}'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def plot_saliency_map(model, x, y):
    # Find the index of the to be visualized layer above
    layer_index = utils.find_layer_idx(model, 'dense_3')

    # Swap softmax with linear to get better results
    model.layers[layer_index].activation = keras.activations.linear
    model = utils.apply_modifications(model)

    # Calculate saliency_map and visualize it
    saliency = np.zeros((512, 512))
    m = len(x)

    for i in range(m):  # Get input
        input_image = x[i]
        input_class = y[i]  # Matplotlib preparations
        saliency += visualize_saliency(model, layer_index, filter_indices=input_class, seed_input=input_image)

    saliency /= m

    fig = plt.figure(dpi=160)
    cax = plt.imshow((saliency / saliency.max() * 255).astype(np.uint8), cmap='jet')
    cbar = fig.colorbar(cax, ticks=[0, 110, 220])
    cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
    plt.show()


def plot_tsne(model, x, y):
    # First apply PCA to reduce to 30 dims
    pca = PCA(n_components=30)

    # Then TSNE to reduce to 2 dims with 1000 iterations and learning rate of 200
    tsne = TSNE(n_components=2, n_iter=1000, learning_rate=200)

    # Get the output of layer 'dense_1' (1024 features) to reduce the dimension of that output
    layer_name = 'dense_1'
    intermediate_output = model.get_layer(layer_name).output
    print(intermediate_output)
    f = keras.backend.function([model.input], [intermediate_output])
    # Get the features generated when passing X data
    features = np.array(f([x]))
    features = features.squeeze(0)
    print(features.shape)
    # Apply PCA and t-SNE
    pca_result = pca.fit_transform(features)
    tsne_result = tsne.fit_transform(pca_result)
    # Prepare data to be visualized
    tsne_data = dict()
    tsne_data['tsne-2d-one'] = tsne_result[:, 0]
    tsne_data['tsne-2d-two'] = tsne_result[:, 1]
    tsne_data['y'] = get_class(y)

    # Visualize the data reduced to 2 dimensions
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.hls_palette(3, l=.6, s=.7),
        data=tsne_data,
        legend="full",
        alpha=0.3
    )
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", default=None, help="path to the directory where the images are stored")
    ap.add_argument("-m", "--model", default=None, help="path to the file where the model is stored")
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
        print("You must specify the file where the model is stored (see help).")
        return

    # Load the model architecture and its weights
    model = load_model(model_file)

    # Load the data
    x_train, y_train, x_test, y_test = load_data(base_dir)

    # Convert data to numpy array
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train2 = y_train
    y_test2 = y_test

    # Binarize y data
    y_test = to_categorical(y_test)

    print('Plotting ROC curve...')
    plot_roc_curve(model, x_test, y_test)

    print('Plotting t-SNE...')
    plot_tsne(model, x_train, y_train2)

    print('Plotting saliency map...')
    plot_saliency_map(model, x_test, y_test2)


if __name__ == '__main__':
    """ 
    Match input image or current life video feed with the selected template
    """
    # GPU memory growth and just use GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only see the gpu 0
    main()
