# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This code was adapted from the file heatmap.py from the project ThirdEye, a misbehaviour predictor for autonomous
# vehicles developed within the ERC project PRECRIME and released under the "MIT License Agreement".
# Please see the LICENSE file for further information.
import os
import random
import shutil
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import image as mpimg
from natsort import natsorted
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import Scorecam


from tqdm import tqdm

from utils.dataset_utils import preprocess

sys.setrecursionlimit(10000)
import config
from global_log import GlobalLog

from PIL import Image


logger = GlobalLog("heatmap_generator")


def score_when_decrease(output):
    #return -1.0 * output[:, 0]
    return output[:, 0]


def compute_heatmap(simulation_name: str, attention_type: str = "SmoothGrad"):
    logger.debug("Computing attention heatmaps for simulation %s using %s" % (simulation_name, attention_type))

    # load vit_model paths from filesystem
    image_names_list = natsorted(glob(os.path.join(os.getcwd(), simulation_name, '*.jpg')))
    image_list = list(map(Image.open, natsorted(glob(os.path.join(os.getcwd(), simulation_name, '*.jpg')))))

    assert len(image_names_list) != 0
    assert len(image_list) != 0

    logger.debug("read %d vit_model from file" % len(image_list))

    index_step = 10
    image_names_list = image_names_list[::index_step]
    image_list = image_list[::index_step]

    logger.debug(f"get every {index_step}th element starting at index 0; new size is {len(image_list)}")

    # load self-driving car model
    self_driving_car_model = load_model(Path(os.path.join('models', 'udacity-dave2.h5')))

    # load attention model
    saliency = None
    if attention_type == "SmoothGrad":
        saliency = Saliency(self_driving_car_model, model_modifier=None)
    elif attention_type == "GradCam++":
        saliency = GradcamPlusPlus(self_driving_car_model, model_modifier=None)
    elif attention_type == "Faster-ScoreCAM":
        saliency = Scorecam(self_driving_car_model, model_modifier=None)

    avg_heatmaps = []
    avg_gradient_heatmaps = []
    list_of_image_paths = []
    prev_hm = gradient = np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    # create directory for the heatmaps
    path_save_heatmaps = os.path.join(os.getcwd(),
                                      simulation_name,
                                      "heatmaps-" + attention_type.lower(),
                                      "IMG")
    if os.path.exists(path_save_heatmaps):
        logger.debug("Deleting folder at {}".format(path_save_heatmaps))
        shutil.rmtree(path_save_heatmaps)
    logger.debug("Creating image folder at {}".format(path_save_heatmaps))
    os.makedirs(path_save_heatmaps)

    for idx, img in enumerate(tqdm(image_list)):

        # x = np.asarray(img).astype('float32')
        x = preprocess(np.asarray(img))
        x = x.astype('float32')
        # print(x)
        # compute heatmap image
        saliency_map = None
        if attention_type == "SmoothGrad":
            saliency_map = saliency(score_when_decrease, x, smooth_samples=20, smooth_noise=0.20)
        elif attention_type == "GradCam++":
            saliency_map = saliency(score_when_decrease, x, penultimate_layer=-1)
        elif attention_type == "Faster-ScoreCAM":
            saliency_map = saliency(score_when_decrease,
                                    x,
                                    penultimate_layer=-1,
                                    max_N=10)

        # compute average of the heatmap
        average = np.average(saliency_map)

        # compute gradient of the heatmap
        if idx == 0:
            gradient = 0
        else:
            gradient = abs(prev_hm - saliency_map)
        average_gradient = np.average(gradient)
        prev_hm = saliency_map

        # store the heatmaps
        img = image_names_list[idx]
        file_name = img.split('/')[-1]
        file_name = "htm-" + attention_type.lower() + '-' + file_name
        path_name = os.path.join(path_save_heatmaps, file_name)
        mpimg.imsave(path_name, np.squeeze(saliency_map))

        list_of_image_paths.append(path_name)

        avg_heatmaps.append(average)
        avg_gradient_heatmaps.append(average_gradient)

    # return the index of the heatmap having the highest gradient score
    # TODO: refine to take sequence into account
    print(f"avg_heatmaps {avg_heatmaps}")
    min_avg_heatmaps = avg_heatmaps.index(np.min(avg_heatmaps))
    print(f"min_avg_heatmaps {min_avg_heatmaps}")

    print(f"avg_gradient_heatmaps {avg_gradient_heatmaps}")
    max_gradient_score_idx = avg_gradient_heatmaps.index(np.max(avg_gradient_heatmaps))
    print(f"max_gradient_score_idx {max_gradient_score_idx}")

    direction = mutation_direction(heat_map=saliency_map[max_gradient_score_idx])
    return min_avg_heatmaps*index_step, direction


def mutation_direction(heat_map = None):
    assert heat_map is not None, "heat_map can not be none"
    assert isinstance(heat_map, np.ndarray), "heat_map must be a numpy array"
    assert heat_map.ndim == 2, "heat_map must be a 2D image"
    middle_index = heat_map.shape[1]//2

    left_half = heat_map[:, :middle_index]
    right_half = heat_map[:, middle_index:]

    left_mean = np.mean(left_half)
    right_mean = np.mean(right_half)

    print(f"left half mean {left_mean} \nright half mean {right_mean}")
    direction = "R" if left_mean < right_mean else "L"   # to reduce to high attention part in heat map

    return direction


if __name__ == '__main__':
    '''
    Given a simulation by Udacity, the script reads the corresponding image paths from the csv and creates a heatmap for
    each driving image. The heatmap is created with the SmoothGrad algorithm available from tf-keras-vis 
    (https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html#SmoothGrad). The scripts generates a separate 
    IMG/ folder and csv file.
    '''

    compute_heatmap(
        simulation_name='simulations/24-01-18-21-23-XAI-seed=21-num-episodes=8-agent=supervised-num-control-nodes=9-max-angle=70/episode0',
        attention_type='GradCam++')
