import os
import random
import shutil
import sys
from pathlib import Path

import config
import numpy as np
from alibi.explainers import IntegratedGradients
from global_log import GlobalLog
from keras.models import load_model

# import keras
from PIL import Image
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import Scorecam

# from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

logger = GlobalLog("heatmap_generator")
sys.setrecursionlimit(10000)

import csv

from matplotlib import image as mpimg
from utils.dataset_utils import preprocess

# import pandas as pd
# pd.options.mode.chained_assignment = None


def mutation_direction(heat_map=None):
    assert heat_map is not None, "heat_map can not be none"
    assert isinstance(heat_map, np.ndarray), "heat_map must be a numpy array"
    assert heat_map.ndim == 2, "heat_map must be a 2D image"
    middle_index = heat_map.shape[1] // 2

    left_half = heat_map[:, :middle_index]
    right_half = heat_map[:, middle_index:]

    left_mean = np.mean(left_half)
    right_mean = np.mean(right_half)

    logger.debug(f"left half mean {left_mean} \nright half mean {right_mean}")
    direction = (
        "R" if left_mean < right_mean else "L"
    )  # to reduce to high attention part in heat map

    # return High attention side
    return direction


class AttentionManager:
    def __init__(
        self, simulation_name: str, index_step=10, attention_type: str = "Faster-ScoreCAM"
    ):
        self.score = CategoricalScore(0)  # regression model only have one output
        # self.replace2linear = ReplaceToLinear() # no softmax for regression model

        self.model = load_model(Path(os.path.join("models", "udacity-dave2.h5")))
        self.attention_type = attention_type

        # make heat map dataframe
        # self.df_heat_map = pd.read_csv(simulation_name + '.csv')

        self.img_paths = []
        self.avg_gradient_heatmaps = []
        self.closest_cps = []
        self.frame_ids = []

        with open(simulation_name + ".csv", "r") as f:
            reader = csv.reader(f)
            # skip the first line
            next(reader)
            for row in reader:
                if int(row[0]) % index_step == 0:
                    self.img_paths.append(row[1])
                    img_info = row[1].split("/")[-1][3:-4].split("-")
                    self.frame_ids.append(int(img_info[0]))
                    self.closest_cps.append(int(img_info[-1]))
        self.closest_cps = np.array(self.closest_cps)
        assert len(self.img_paths) != 0
        # self.df_heat_map = self.df_heat_map.iloc[::index_step][["img", "closest_cp"]]
        # self.df_heat_map['id'] = self.df_heat_map.index
        # self.df_heat_map = self.df_heat_map.reset_index(drop=True)

        logger.debug(
            f"get every {index_step}th element starting at index 0; new size is {len(self.img_paths) }"
        )
        # self.avg_gradient_heatmaps= None
        # self.df_heat_map["avg_heatmaps"] = None
        # self.df_heat_map["heat_map_path"] = None

        if simulation_name is not None:
            self.heat_map_directory(simulation_name)

    def heat_map_directory(self, simulation_name):
        # create directory for the heatmaps
        self.path_save_heatmaps = os.path.join(
            os.getcwd(), simulation_name, "heatmaps-" + self.attention_type.lower(), "IMG"
        )
        if os.path.exists(self.path_save_heatmaps):
            logger.debug("Deleting folder at {}".format(self.path_save_heatmaps))
            shutil.rmtree(self.path_save_heatmaps)
        logger.debug("Creating image folder at {}".format(self.path_save_heatmaps))
        os.makedirs(self.path_save_heatmaps)

    def preprocess_image(self):
        # load vit_model paths from filesystem and preprocess the vit_model
        image_list = list(map(Image.open, self.img_paths))

        assert len(image_list) != 0

        logger.debug("read %d vit_model from file" % len(image_list))

        processed_images = []

        for image in image_list:
            img = preprocess(np.asarray(image), if_yuv=True)
            processed_images.append(img.astype("float32"))
        processed_images = np.asarray(processed_images)

        return processed_images

    def compute_attention_maps(self):  # vit_model should have the shape: (x, 28, 28) where x>=1
        # retrieve the control point which is closest to the hottest point

        prev_hm = gradient = np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

        X = self.preprocess_image()
        switch = {
            "VanillaSaliency": self.vanilla_saliency,
            "SmoothGrad": self.smooth_grad,
            "GradCAM": self.grad_cam,
            "GradCAM++": self.grad_cam_pp,
            "ScoreCAM": self.score_cam,
            "Faster-ScoreCAM": self.faster_score_cam,
            "IntegratedGradients": self.integrated_gradients,
        }
        attention_maps = switch.get(self.attention_type)(X)

        for idx, ht_map in enumerate(attention_maps):
            # compute average of the heatmap
            img_path = self.img_paths[idx]

            # self.df_heat_map["avg_heatmaps"][idx] = np.average(ht_map)

            # compute gradient of the heatmap
            if idx == 0:
                gradient = 0
            else:
                gradient = abs(prev_hm - ht_map)
            prev_hm = ht_map
            self.avg_gradient_heatmaps.append(np.average(gradient))

            # store the heatmaps

            file_name = img_path.split("/")[-1]
            file_name = "htm-" + self.attention_type.lower() + "-" + file_name
            path_name = os.path.join(self.path_save_heatmaps, file_name)
            mpimg.imsave(path_name, np.squeeze(ht_map))
            # self.df_heat_map["heat_map_path"][idx] = path_name

        # print(f"avg_heatmaps {self.df_heat_map[['avg_heatmaps', 'avg_gradient_heatmaps', 'closest_cp']]}")
        # direction = mutation_direction(heat_map=attention_maps[min_avg_heatmaps])
        # return min_avg_heatmaps*self.index_step, direction
        self.avg_gradient_heatmaps = np.array(self.avg_gradient_heatmaps)
        cp_id = self.control_point_selection()
        # print(self.closest_cps == cp_id)
        attention_maps = np.array(attention_maps)
        # print(attention_maps.shape)
        # print(np.mean(attention_maps[self.closest_cps == cp_id], axis=0).shape)
        direction = mutation_direction(np.mean(attention_maps[self.closest_cps == cp_id], axis=0))
        print(direction)
        return cp_id, direction

    def control_point_selection(self, factor=40):
        unique_cp = np.unique(self.closest_cps)

        avg_gradient_heatmaps_by_cp = np.array(
            [np.mean(self.avg_gradient_heatmaps[self.closest_cps == cp]) for cp in unique_cp]
        )
        weight = avg_gradient_heatmaps_by_cp / np.sum(avg_gradient_heatmaps_by_cp) * factor
        weight = np.exp(weight) / np.sum(np.exp(weight))

        cp_id = random.choices(population=unique_cp, weights=weight, k=1)[0]
        logger.debug(f"select {cp_id} based on the weights {weight}")
        # cp_id = random.choices(population=unique_cp, k=1)[0]
        return cp_id

    def single_attention_map(self, preprocessed_image, attention_type="Faster-ScoreCAM"):
        preprocessed_image = preprocessed_image.astype("float32")
        switch = {
            "VanillaSaliency": self.vanilla_saliency,
            "SmoothGrad": self.smooth_grad,
            "GradCAM": self.grad_cam,
            "GradCAM++": self.grad_cam_pp,
            "ScoreCAM": self.score_cam,
            "Faster-ScoreCAM": self.faster_score_cam,
            "IntegratedGradients": self.integrated_gradients,
        }
        attention_map = switch.get(attention_type)(preprocessed_image)
        return attention_map

    def vanilla_saliency(self, X):
        # Create Saliency object.
        saliency = Saliency(self.model, model_modifier=None, clone=True)
        # Generate saliency map
        saliency_map = saliency(self.score, X)
        return saliency_map

    def smooth_grad(self, X):
        # Create Saliency object.
        saliency = Saliency(self.model, model_modifier=None, clone=True)

        # Generate saliency map with smoothing that reduce noise by adding noise
        saliency_map = saliency(
            self.score,
            X,
            smooth_samples=20,  # The number of calculating gradients iterations.
            smooth_noise=0.20,
        )  # noise spread level.
        return saliency_map

    def grad_cam(self, X):
        # Create Gradcam object
        gradcam = Gradcam(self.model, model_modifier=None, clone=True)

        # Generate heatmap with GradCAM
        cam = gradcam(self.score, X, penultimate_layer=-1)
        return cam

    def grad_cam_pp(self, X):
        # Create GradCAM++ object
        gradcam = GradcamPlusPlus(self.model, model_modifier=None, clone=True)

        # Generate heatmap with GradCAM
        cam = gradcam(self.score, X, penultimate_layer=-1)
        return cam

    def score_cam(self, X):
        # Create ScoreCAM object
        scorecam = Scorecam(self.model)

        # Generate heatmap with ScoreCAM
        cam = scorecam(self.score, X, penultimate_layer=-1)
        return cam

    def faster_score_cam(self, X):
        # Create ScoreCAM object
        scorecam = Scorecam(self.model, model_modifier=None)

        # Generate heatmap with Faster-ScoreCAM
        cam = scorecam(self.score, X, penultimate_layer=-1, max_N=20)
        return cam

    def integrated_gradients(self, X, steps=20):
        ig = IntegratedGradients(self.model, n_steps=steps, method="gausslegendre")
        # predictions = self.model(X).numpy().argmax(axis=1)
        """predictions = []
        for img in X:
            img = img.astype(int)
            predictions.append(self.model.predict(img.reshape(-1, 160, 320, 3), batch_size=1, verbose=0))"""

        # baseline grassland
        grass_baseline = np.zeros((160, 320, 3))
        grass_baseline[:, :, 0:3] = [108.85083333, 133.98472222, 93.38805556]
        # grass_baseline = grass_baseline / 255.0
        grass_baseline = grass_baseline[np.newaxis, ...]

        explanation = ig.explain(X, baselines=grass_baseline, target=0)

        attributions = explanation.attributions[0]
        # remove single-dimensional shape of the array.
        # attributions = attributions.squeeze()
        attributions = np.abs(attributions)

        attrs_mean = attributions.mean(axis=3)

        normalized_attributions = np.zeros(shape=attrs_mean.shape)

        # Normalization
        for i in range(attrs_mean.shape[0]):
            try:
                # print(f"attention map difference {np.max(attributions[i]) - np.min(attributions[i])}")
                normalized_attributions[i] = (attrs_mean[i] - np.min(attrs_mean[i])) / (
                    np.max(attrs_mean[i]) - np.min(attrs_mean[i])
                )
            except ZeroDivisionError:
                print("Error: Cannot divide by zero")
                return

        return normalized_attributions


if __name__ == "__main__":
    episode_sub_path = "simulations/03-15-21-52-RANDOM-seed=0/episode15"
    episode_sub_path2 = "simulations/03-15-22-02-RANDOM-seed=1/episode10"
    episode_sub_path3 = "simulations/03-15-22-08-RANDOM-seed=2/episode13"
    episode_sub_path4 = "simulations/03-15-22-18-RANDOM-seed=1/episode14"
    episode_sub_path5 = "simulations/03-15-22-24-RANDOM-seed=2/episode13"
    episode_sub_path6 = "simulations/03-16-18-35-XAI-seed=8/episode27"

    episode_sub_path7 = "simulations/03-18-14-41-XAI-seed=2/episode1"

    attention_manager = AttentionManager(
        simulation_name=episode_sub_path7, index_step=10, attention_type="Faster-ScoreCAM"
    )
    attention_manager.compute_attention_maps()
    # attention_manager.control_point_selection()

    # pop.evaluate_population(0)
