import random
import tensorflow as tf
import numpy as np
from config import SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS
import gzip

def load_mnist_test(popsize, number):
    file_test_x = './original_dataset/t10k-images-idx3-ubyte.gz'
    file_test_y = './original_dataset/t10k-labels-idx1-ubyte.gz'

    with gzip.open(file_test_x, 'rb') as f:
        _ = np.frombuffer(f.read(16), dtype=np.uint8, count=4)
        images = np.frombuffer(f.read(), dtype=np.uint8)
        test_x = images.reshape(-1, 28, 28)

    with gzip.open(file_test_y, 'rb') as f:
        _ = np.frombuffer(f.read(8), dtype=np.uint8, count=2)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        test_y = labels

    idx = [i for i, label in enumerate(test_y) if label == number]
    #print(f"number of {number} is {len(idx)}")
    filtered_test_y = test_y[idx]
    filtered_test_x = test_x[idx]

    if popsize < filtered_test_y.shape[0]:
        select_index = np.random.choice(range(filtered_test_x.shape[0]), size=popsize, replace=False)
        select_index = np.sort(select_index)
        # print(f"select index {select_index}")
        return filtered_test_x[select_index], filtered_test_y[select_index]
    else:
        return filtered_test_x, filtered_test_y


def set_all_seeds(digit):
    random.seed(SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS[digit])
    np.random.seed(SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS[digit])
    tf.keras.utils.set_random_seed(SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS[digit])

