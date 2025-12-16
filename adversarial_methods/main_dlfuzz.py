# -*- coding: utf-8 -*-
from PIL import Image
import os

# from tqdm import tqdm
#
# from tensorflow.keras.layers import Input, Activation
from tensorflow.keras import backend as K

# from tensorflow import keras
from adversarial_methods.DLFuzz.dlfuzz import DLFuzz
from adversarial_methods.DLFuzz.utils import clear_up_dir, deprocess_image
from utils import set_all_seeds, load_mnist_test
from config import POPSIZE
from predictor import Predictor
import csv
import numpy as np

if __name__ == "__main__":
    pop_size = POPSIZE
    model = Predictor.model

    # output images
    save_dir = "./result/dlfuzz_digits/"
    clear_up_dir(save_dir)
    csv_file = "./result/dlfuzz_digits/record_dlfuzz.csv"

    # prepare
    K.set_learning_phase(0)
    dlfuzz = DLFuzz(model)

    # start
    for digit in range(10):  # range(10):
        set_all_seeds(digit)
        x_test, y_test = load_mnist_test(pop_size, digit)

        for i, (tmp_img, image_label) in enumerate(zip(x_test, y_test)):

            tmp_img = tmp_img.astype("float32") / 255
            prediction, confidence = Predictor.predict_single(
                np.expand_dims(tmp_img, 0), image_label
            )
            if prediction != image_label:
                print(f"drop id {i}, label {image_label}, predict {prediction, confidence}")
            else:
                # calculate fuzz image
                gen_img = dlfuzz.generate_adversarial_image(tmp_img.reshape((28, 28, 1)))
                prediction, confidence = Predictor.predict_single(
                    np.expand_dims(gen_img, 0), image_label
                )
                if prediction != image_label:
                    print(
                        f"succeed misclass id {i}, label {image_label}, predict {prediction, confidence}"
                    )
                    save_path = os.path.join(save_dir, f"digit_{digit}")
                    os.makedirs(save_path, exist_ok=True)
                    save_img = os.path.join(save_path, f"id_{i}.png")

                    gen_img_deprocessed = deprocess_image(gen_img)
                    Image.fromarray(gen_img_deprocessed).save(save_img)
                    # record csv
                    if os.path.exists(csv_file):
                        with open(csv_file, "a", newline="") as f:
                            csv_writer = csv.writer(f)
                            csv_writer.writerow(
                                [digit, i, image_label, prediction, confidence, save_img]
                            )
                    else:
                        with open(csv_file, "w", newline="") as f:
                            csv_writer = csv.writer(f)
                            csv_writer.writerow(
                                [
                                    "digit",
                                    "id",
                                    "expected_label",
                                    "predicted_label",
                                    "confidence",
                                    "image_path",
                                ]
                            )
                            csv_writer.writerow(
                                [digit, i, image_label, prediction, confidence, save_img]
                            )
