from global_log import GlobalLog
import numpy as np
import random

import tensorflow as tf


def set_random_seed(seed):

    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # Seed tensorflow RNG
    # tf.random.set_seed(seed)

    tf.keras.utils.set_random_seed(seed)

