import random
from os import makedirs
from os.path import exists, basename, join

from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

from folder import Folder
from config import IMG_SIZE, SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS
import numpy as np


# load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


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


def set_all_seeds(digit):
    random.seed(SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS[digit])
    np.random.seed(SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS[digit])
    tf.keras.utils.set_random_seed(SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS[digit])


def closest_2d_point(target_point, points_list):
    # Convert the points list to a NumPy array for easier computation
    points_array = np.array(points_list)

    # Calculate the Euclidean distance between the target point and all points in the list
    distances = np.linalg.norm(points_array - target_point, axis=1)

    # Find the index of the point with the minimum distance
    closest_index = np.argmin(distances)

    # Retrieve the closest point
    closest_point = points_list[closest_index]

    return closest_point