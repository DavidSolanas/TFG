"""
    File: show_results.py
    Author: David Solanas Sanz
    TFG
"""

import os
import argparse
import matplotlib.pyplot as plt
from itertools import cycle, product
from sklearn.metrics import roc_curve, auc
import numpy as np
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from vis.visualization import visualize_saliency
from vis.utils import utils
import keras
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from sklearn import metrics
from keras.utils import to_categorical
from skimage.transform import resize


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


def plot_roc_curve(model, y_score, y):
    # Plot linewidth.
    lw = 2
    n_classes = 3
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
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
        plt.plot(1 - fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC for {0}: AUC = {1:0.2f}'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([1.02, -0.02], [-0.02, 1.02], 'k--', lw=lw)
    plt.xlim([1.02, -0.02])
    plt.ylim([-0.02, 1.02])
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
        print(i)
        input_image = x[i]
        input_class = y[i]  # Matplotlib preparations
        saliency += visualize_saliency(model, layer_index, filter_indices=input_class, seed_input=input_image)

    saliency /= m

    fig = plt.figure()
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

    intermediate_model = keras.Model(inputs=model.input, outputs=intermediate_output)
    intermediate_model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                               loss='categorical_crossentropy',
                               metrics=['acc'])

    # Get the features generated when passing X data

    features = intermediate_model.predict(x)
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    # plt.tight_layout()


def plot_cm(y_test, y_pred, normalize=False):
    class_names = ['AD', 'CN', 'MCI']
    n_classes = 3
    y_prd = [np.argmax(y) for y in y_pred]
    for i in range(n_classes):
        y_score = [y == i for y in y_prd]
        y_score = np.array(y_score).astype(int)
        y_true = y_test[:, i]
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_score).ravel()
        specificity = tn / (tn + fp)
        sensitivity = metrics.recall_score(y_true, y_score)  # tp / (tp + fn)
        precision = metrics.precision_score(y_true, y_score)
        f1_score = metrics.f1_score(y_true, y_score)
        print('############################################')
        print('Metrics for class {}'.format(class_names[i]))
        print('Sensitivity: ', sensitivity)
        print('Specificity: ', specificity)
        print('Precision: ', precision)
        print('F1-Score: ', f1_score)
        print('TP: ', tp)
        print('TN: ', tn)
        print('FP: ', fp)
        print('FN: ', fn)
        print()

    '''precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    recall = sensitivity
    f1_score = 2 * precision * recall / (precision + recall)

    print('Sensitivity AD class: %.2f' % sensitivity[0])
    print('Sensitivity CN class: %.2f' % sensitivity[1])
    print('Sensitivity MCI class: %.2f' % sensitivity[2])
    print('Specificity AD class: %.2f' % specificity[0])
    print('Specificity CN class: %.2f' % specificity[1])
    print('Specificity MCI class: %.2f' % specificity[2])
    print('Precision AD class: %.2f' % precision[0])
    print('Precision CN class: %.2f' % precision[1])
    print('Precision MCI class: %.2f' % precision[2])
    print('F1-Score AD class: %.2f' % f1_score[0])
    print('F1-Score CN class: %.2f' % f1_score[1])
    print('F1-Score MCI class: %.2f' % f1_score[2])

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          normalize=normalize)
    '''


def standarize(x):
    return (x - np.mean(x)) / np.std(x)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


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

    train_datagen = ImageDataGenerator(
        # rotation_range=8,
        # shear_range=np.pi / 16,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.08,
        horizontal_flip=False,
        vertical_flip=False,
    )

    test_datagen = ImageDataGenerator()

    # Set the batch size and calculate the number of steps per epoch
    input_size = 512
    batch_size = 8

    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    print(test_generator.class_indices)
    nb_train_samples = len(train_generator.filenames)
    nb_test_samples = len(test_generator.filenames)
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
    for x_batch, y_batch in test_generator:
        for i in range(len(y_batch)):  # Get input
            x_test.append(x_batch[i])
            y_test.append(y_batch[i])
        batches += 1
        if batches >= nb_test_samples / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

    print(test_generator.classes)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train2 = np.argmax(y_train, axis=1)
    y_test2 = np.argmax(y_test, axis=1)

    # Get the score of the model with test dataset
    _, train_accuracy = model.evaluate(x_train, y_train, batch_size=batch_size)
    _, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_accuracy, test_accuracy))

    y_pred = model.predict(x_test)
    # print(y_pred)
    plot_cm(y_test, y_pred)

    print('Plotting ROC curve...')
    plot_roc_curve(model, y_pred, y_test)

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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    main()
