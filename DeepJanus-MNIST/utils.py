from os import makedirs
from os.path import exists, basename, join

from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from folder import Folder
from config import IMG_SIZE, DATASET
import numpy as np

from predictor import Predictor
import vectorization_tools
import rasterization_tools
import os
import glob
import cv2


# # load the MNIST dataset
# mnist = keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()


def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def print_archive(archive):
    dst = Folder.DST_ARC + "_DJ"
    if not exists(dst):
        makedirs(dst)
    for i, ind in enumerate(archive):
        filename1 = join(dst, basename(
            'archived_' + str(i) +
            '_mem1_l_' + str(ind.m1.predicted_label) +
            '_seed_' + str(ind.seed)))
        plt.imsave(filename1, ind.m1.purified.reshape(28, 28),
                   cmap=cm.gray,
                   format='png')
        np.save(filename1, ind.m1.purified)
        assert (np.array_equal(ind.m1.purified,
                               np.load(filename1 + '.npy')))

        filename2 = join(dst, basename(
            'archived_' + str(i) +
            '_mem2_l_' + str(ind.m2.predicted_label) +
            '_seed_' + str(ind.seed)))
        plt.imsave(filename2, ind.m2.purified.reshape(28, 28),
                   cmap=cm.gray,
                   format='png')
        np.save(filename2, ind.m2.purified)
        assert (np.array_equal(ind.m2.purified,
                               np.load(filename2 + '.npy')))


def print_archive_experiment(archive):
    for i, ind in enumerate(archive):
        digit = ind.m1
        digit.export(ind.id)
        digit = ind.m2
        digit.export(ind.id)
        ind.export()


# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        v = v.reshape(v.shape[0], 1, IMG_SIZE, IMG_SIZE)
    else:
        v = v.reshape(v.shape[0], IMG_SIZE, IMG_SIZE, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v


def input_reshape(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    else:
        x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0
    return x_reshape


def set_all_seeds(seed):
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_data():
    import h5py
    # Load the dataset.
    hf = h5py.File(DATASET, 'r')
    x_test = hf.get('xn')
    x_test = np.array(x_test)
    y_test = hf.get('yn')
    y_test = np.array(y_test)
    return x_test, y_test


def load_icse_data(confidence_is_100, label):
    if confidence_is_100:
        dataset_path = os.path.join("original_dataset/final-mnist/100/", str(label))
    else:
        dataset_path = os.path.join("original_dataset/final-mnist/not_100/", str(label))
    # List all files in the directory
    image_files = glob.glob(dataset_path + '/*.png')
    image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    # Initialize an empty list to store the images
    images = []

    # Load each image and append to the list
    for image_path in image_files:
        # Load image in grayscale mode
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    # Convert list of images to a single 3D numpy array (matrix)
    x_test = np.array(images)

    # print(x_test.shape)
    y_test = np.array([label] * len(x_test))
    return x_test, y_test, image_files



def check_label(x_test, y_test, image_files, explabel):
    predictions, _ = (Predictor.predict(img=x_test, label=y_test))
    predictions = np.array(predictions)

    # drop labels that are not EXPLABEL
    data_size = x_test.shape[0]

    x_test = x_test[predictions == explabel]
    y_test = y_test[predictions == explabel]
    image_files = image_files[predictions == explabel]

    if data_size != x_test.shape[0]:
        print("Dropped {} images with wrong label".format(data_size - x_test.shape[0]))
    data_size = x_test.shape[0]
    assert data_size != 0, "No data left"

    # drop labels that are not EXPLABEL after rasterization
    new_predictions = []
    for img in x_test:
        xml_desc = vectorization_tools.vectorize(img)
        rasterized = rasterization_tools.rasterize_in_memory(xml_desc)
        prediction_rasterized, _ = Predictor.predict_single(rasterized, explabel)
        new_predictions.append(prediction_rasterized)

    new_predictions = np.array(new_predictions)
    x_test = x_test[np.where(new_predictions == explabel)]
    y_test = y_test[np.where(new_predictions == explabel)]
    image_files = image_files[np.where(new_predictions == explabel)]

    if data_size != x_test.shape[0]:
        print("Dropped {} images after rasterization".format(data_size - x_test.shape[0]))
    assert x_test.shape[0] != 0, "No data left"

    return x_test, y_test, image_files
