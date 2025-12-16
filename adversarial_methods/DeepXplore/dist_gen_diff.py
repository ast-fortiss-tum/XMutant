"""
DeepXplore: https://github.com/peikexin9/deepxplore
"""

from __future__ import print_function

import argparse
import random

import h5py
from keras.datasets import mnist
from keras.layers import Input
from keras.utils import to_categorical

# from Model1 import Model1
# from Model2 import Model2
# from Model3 import Model3
# from Model4 import Model4
from utils import *
import imageio
import numpy as np
import math
import time
import datetime

# random.seed(3)

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description="Main function for difference-inducing input generation in MNIST dataset"
)
parser.add_argument(
    "transformation", help="realistic transformation type", choices=["light", "occl", "blackout"]
)
parser.add_argument(
    "weight_diff", help="weight hyperparm to control differential behavior", type=float
)
parser.add_argument("weight_nc", help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument("weight_vae", help="weight hyperparm to control vae goal", type=float)
parser.add_argument("step", help="step size of gradient descent", type=float)
parser.add_argument("seeds", help="number of seeds of input", type=int)
parser.add_argument("grad_iterations", help="number of iterations of gradient descent", type=int)
parser.add_argument("threshold", help="threshold for determining neuron activated", type=float)
parser.add_argument(
    "-t",
    "--target_model",
    help="target model that we want it predicts differently",
    choices=[0, 1, 2, 3],
    default=0,
    type=int,
)
parser.add_argument(
    "-sp",
    "--start_point",
    help="occlusion upper left corner coordinate",
    default=(0, 0),
    type=tuple,
)
parser.add_argument(
    "-occl_size", "--occlusion_size", help="occlusion size", default=(10, 10), type=tuple
)
args = parser.parse_args()

DATASET = "/home/vin/PycharmProjects/dola/DistributionAwareDNNTesting/MNIST/original_dataset/janus_dataset_comparison.h5"

print("\n\n")

output_directory = (
    "./empstudy/generated_inputs_Model"
    + str(args.target_model + 1)
    + "/"
    + (args.transformation)
    + "/"
)

# Create directory to store generated tests
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
delete_files_from_dir(output_directory, "png")

# Create a subdirectory inside output directory
# for saving original seed images used for test generation
orig_directory = output_directory + "seeds/"
if not os.path.exists(orig_directory):
    os.makedirs(orig_directory)
delete_files_from_dir(orig_directory, "png")

# input image dimensions
img_rows, img_cols = 28, 28
img_dim = img_rows * img_cols
# the data, shuffled and split between train and test sets
# (_, _), (x_test, y_test) = mnist.load_data()

# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# x_test = x_test.astype('float32')
# x_test /= 255


# Load the dataset.
# hf = h5py.File(DATASET, 'r')
# x_test = hf.get('xn')
# x_test = np.array(x_test)
# y_test = hf.get('yn')
# y_test = np.array(y_test)
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# x_test = x_test.astype('float32')
# x_test /= 255
# seeds = x_test


# Fetch the starting seeds from file
# starting_seeds = [i for i in range(len(y_test))]
# random.shuffle(starting_seeds)
# starting_seeds = starting_seeds[:args.seeds]


CLASS = 5
(_, _), (x_test, y_test) = mnist.load_data()

idxs = np.argwhere(y_test == CLASS)
starting_seeds = x_test[idxs]
starting_seeds = starting_seeds.reshape(starting_seeds.shape[0], 28, 28, 1)
starting_seeds = starting_seeds.astype("float32")
starting_seeds = starting_seeds / 255.0

random.shuffle(starting_seeds)
img_paths = starting_seeds

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(
    model1, model2, model3
)

# ==============================================================================================
# start gen inputs

start_time = datetime.datetime.now()
# seed_nums = np.load('./seeds/seeds_'+str(args.seeds)+'.npy')
result_loop_index = []
result_coverage = []
loop_index = 0

print("Started exploration")

POPSIZE = 34
SEEDCOUNT = 0
adversial_num = 0
# TODO: changed source
# for current_seed in seed_nums:
while SEEDCOUNT < len(img_paths) and adversial_num < POPSIZE:
    # for idx, current_seed in enumerate(starting_seeds):
    # below logic is to track number of iterations under progress
    print("Loop number: " + str(loop_index))
    print("seed count: " + str(SEEDCOUNT))
    loop_index += 1

    # current_seed = starting_seeds[SEEDCOUNT]
    current_seed = img_paths[SEEDCOUNT]
    # gen_img = np.expand_dims(x_test[current_seed], axis=0)
    gen_img = np.expand_dims(current_seed, axis=0)
    orig_img = gen_img.copy()
    SEEDCOUNT += 1
    # first check if input already induces differences
    label1, label2, label3 = (
        np.argmax(model1.predict(gen_img)[0]),
        np.argmax(model2.predict(gen_img)[0]),
        np.argmax(model3.predict(gen_img)[0]),
    )

    if not label1 == label2 == label3:
        print("input already causes different outputs")
        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
        update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

        if args.target_model == 0:
            result_coverage.append(neuron_covered(model_layer_dict1)[2])
        elif args.target_model == 1:
            result_coverage.append(neuron_covered(model_layer_dict2)[2])
        elif args.target_model == 2:
            result_coverage.append(neuron_covered(model_layer_dict3)[2])
        elif args.target_model == 3:
            result_coverage.append(neuron_covered(model_layer_dict3)[2])
        # print('covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
        #      % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
        #         neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
        #         neuron_covered(model_layer_dict3)[2]))
        # averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
        #               neuron_covered(model_layer_dict3)[0]) / float(
        #    neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
        #    neuron_covered(model_layer_dict3)[
        #        1])
        # print('averaged covered neurons %.3f' % averaged_nc)

        gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
        # imageio.imwrite(output_directory + 'already_differ_' + str(SEEDCOUNT) + '_' + str(label1) + '_' + str(label2) + '_' + str(label3) + '.png', gen_img_deprocessed)
        continue

    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(
            model1.get_layer("before_softmax").output[..., orig_label]
        )
        loss2 = K.mean(model2.get_layer("before_softmax").output[..., orig_label])
        loss3 = K.mean(model3.get_layer("before_softmax").output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer("before_softmax").output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(
            model2.get_layer("before_softmax").output[..., orig_label]
        )
        loss3 = K.mean(model3.get_layer("before_softmax").output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer("before_softmax").output[..., orig_label])
        loss2 = K.mean(model2.get_layer("before_softmax").output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(
            model3.get_layer("before_softmax").output[..., orig_label]
        )
    elif args.target_model == 3:
        loss1 = K.mean(model1.get_layer("before_softmax").output[..., orig_label])
        loss2 = K.mean(model2.get_layer("before_softmax").output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(
            model3.get_layer("before_softmax").output[..., orig_label]
        )
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (
        loss1_neuron + loss2_neuron + loss3_neuron
    )

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # Gradient computation of loss with respect to the input
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # function to calculate the loss and grads given the input tensor
    iterate = K.function(
        [input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads]
    )

    # Running gradient ascent
    for iters in range(args.grad_iterations):
        print("iteration number: " + str(iters))
        (
            loss_value1,
            loss_value2,
            loss_value3,
            loss_neuron1,
            loss_neuron2,
            loss_neuron3,
            grads_value,
        ) = iterate([gen_img])
        # Apply domain specific constraints
        if args.transformation == "light":
            grads_value = constraint_light(grads_value)
        elif args.transformation == "occl":
            grads_value = constraint_occl(grads_value, args.start_point, args.occlusion_size)
        elif args.transformation == "blackout":
            grads_value = constraint_black(grads_value)

        # generate the new test input
        gen_img += grads_value * args.step
        gen_img = np.clip(gen_img, 0, 1)
        predictions1 = np.argmax(model1.predict(gen_img)[0])
        predictions2 = np.argmax(model2.predict(gen_img)[0])
        predictions3 = np.argmax(model3.predict(gen_img)[0])

        # if iters == int(args.grad_iterations-1):
        #    gen_img_deprocessed = deprocess_image(gen_img)
        #
        #    # save the result to disk
        #    imageio.imwrite(
        #        output_directory + str(loop_index) + '_' + 'last' + '.png',
        #        gen_img_deprocessed)

        # if not predictions1 == predictions2 == predictions3:
        if not predictions1 == CLASS:
            print("misbehaviour detected")
            # Update coverage
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            # Track the seed numbers and coverage achieved for final result
            result_loop_index.append(loop_index)
            if args.target_model == 0:
                result_coverage.append(neuron_covered(model_layer_dict1)[2])
            elif args.target_model == 1:
                result_coverage.append(neuron_covered(model_layer_dict2)[2])
            elif args.target_model == 2:
                result_coverage.append(neuron_covered(model_layer_dict3)[2])
            elif args.target_model == 3:
                result_coverage.append(neuron_covered(model_layer_dict3)[2])

            np.save(
                output_directory
                + str(loop_index)
                + "_"
                + str(predictions1)
                + "_"
                + str(predictions2)
                + "_"
                + str(predictions3)
                + ".npy",
                gen_img,
            )
            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            # save the result to disk
            imageio.imwrite(
                output_directory
                + str(loop_index)
                + "_"
                + str(predictions1)
                + "_"
                + str(predictions2)
                + "_"
                + str(predictions3)
                + ".png",
                gen_img_deprocessed,
            )
            imageio.imwrite(
                orig_directory
                + str(loop_index)
                + "_"
                + str(predictions1)
                + "_"
                + str(predictions2)
                + "_"
                + str(predictions3)
                + "_orig.png",
                orig_img_deprocessed,
            )

            adversial_num += 1

            break


duration = (datetime.datetime.now() - start_time).total_seconds()
no_tests = len(result_loop_index)

print("No of test inputs generated: ", no_tests)
if no_tests == 0:
    print("Cumulative coverage for tests: 0")
    print("Avg. test generation time: NA s")
else:
    print("Cumulative coverage for tests: ", round(result_coverage[-1], 3))
    print("Avg. test generation time: {} s".format(round(duration / no_tests), 2))
print("Total time: {} s".format(round(duration, 2)))
