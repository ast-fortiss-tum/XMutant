from keras.layers import Input
from keras.utils import to_categorical
from adversarial_methods.DeepXplore.utils import *
import numpy as np
import random
import imageio
from predictor import Predictor
import tensorflow as tf
from utils import set_all_seeds
from population import load_mnist_test
from config import POPSIZE
import csv
from PIL import Image

def init_coverage_tables_single(model1):# (model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    # model_layer_dict2 = defaultdict(bool)
    # model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    # init_dict(model2, model_layer_dict2)
    # init_dict(model3, model_layer_dict3)
    return model_layer_dict1# , model_layer_dict2, model_layer_dict3

# Parameters (replace these with your desired values or use argparse as needed)
transformation = 'occl'  # Options: 'light', 'occl', 'blackout'
weight_diff = 1
weight_nc = 0.1
weight_vae = 0
step = 0.1
seeds = 50
grad_iterations = 2500
threshold = 0.0
target_model = 0
pop_size = POPSIZE

# Prepare data (MNIST dataset example)
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

img_dir = "./result/deepxplore_digits/"
os.makedirs(img_dir, exist_ok=True)
csv_file = "./result/deepxplore_digits/record_deepxplore.csv"

for digit in range(10):  # range(10):
    set_all_seeds(digit)
    x_test, y_test = load_mnist_test(pop_size, digit)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255.0

    # Model setup
    input_tensor = Input(shape=input_shape)
    model = Predictor.model
    model_layer_dict = init_coverage_tables_single(model)

    # Perturbation loop
    for seed_idx, (current_seed,orig_label) in enumerate(zip(x_test,y_test)):
        print(f"Processing seed {seed_idx + 1}/{len(x_test)}")
        gen_img = np.expand_dims(current_seed, axis=0)
        _label = np.argmax(model.predict(gen_img)[0])
        if _label != orig_label:
            print(f"drop original misbehavior seed {seed_idx}")
            break

        layer_name1, index1 = neuron_to_cover(model_layer_dict)
        # Construct loss function
        # ValueError: No such layer: before_softmax. Existing layers are: ['conv2d', 'conv2d_1', 'max_pooling2d', 'dropout', 'flatten', 'dense', 'dropout_1', 'dense_1'].

        gen_img_tensor = tf.convert_to_tensor(gen_img, dtype=tf.float32)  # Convert initial input to tensor
        y = np.zeros((1, 10))
        y[0,orig_label] = 1
        true_labels = tf.convert_to_tensor(y, dtype=tf.float32)
        intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name1).output)

        # gen_img_tensor = tf.Variable(gen_img_tensor)
        # Compute gradients
        # RuntimeError: tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead.
        with tf.GradientTape() as tape:

            tape.watch(gen_img_tensor)
            # Calculate the loss inside the tape context
            # predictions = model(gen_img_tensor, training=False)

            # Loss 1: Difference-based loss
            dense_output = model(gen_img_tensor)
            intermediate_output = intermediate_model(gen_img_tensor)  # Output of the specific layer

            loss1 = -weight_diff * tf.reduce_mean(dense_output[..., orig_label])

            # Loss 2: Neuron activation-based loss

            loss1_neuron = weight_nc * tf.reduce_mean(intermediate_output[..., index1])  # Intermediate layer loss

            # Final loss
            final_loss = loss1 + loss1_neuron  # Use the computed final_loss

        # Compute the gradients of the loss with respect to the input_tensor
        grads = tape.gradient(final_loss, gen_img_tensor)

        # Normalize the gradients
        grads = normalize(grads)

        # iterate = K.function([input_tensor], [loss1, loss1_neuron, grads])

        # Gradient ascent iterations
        for iteration in range(grad_iterations):
            # loss_value1, loss_neuron, grads_value = iterate([gen_img])

            # Apply constraints (transformation-specific)
            if transformation == 'light':
                grads_value = constraint_light(grads)
            elif transformation == 'occl':
                grads_value = constraint_occl(grads, start_point=(0, 0), rect_shape=(10, 10))
            elif transformation == 'blackout':
                grads_value = constraint_black(grads)

            # Update the generated image
            gen_img += grads_value * step
            gen_img = np.clip(gen_img, 0, 1)

            # Check for prediction changes
            new_prediction, confidence = Predictor.predict_single(gen_img, orig_label)
            if new_prediction != orig_label:
                print(f"Misbehavior detected at iteration {iteration}")
                gen_img_deprocessed = deprocess_image(gen_img)
                save_path = os.path.join(img_dir, f"digit_{digit}")
                os.makedirs(save_path, exist_ok=True)
                save_img = os.path.join(save_path, f"id_{seed_idx}.png")
                Image.fromarray(gen_img_deprocessed).save(save_img)
                # record csv
                if os.path.exists(csv_file):
                    with open(csv_file, 'a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([digit,
                                             seed_idx,
                                             orig_label,
                                             new_prediction,
                                             confidence,
                                             iteration,
                                             save_img])
                else:
                    with open(csv_file, 'w', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([
                            'digit',
                            'id',
                            'expected_label',
                            'predicted_label',
                            'confidence',
                            'iteration',
                            'image_path'
                        ])
                        csv_writer.writerow([digit,
                                             seed_idx,
                                             orig_label,
                                             new_prediction,
                                             confidence,
                                             iteration,
                                             save_img])

                break
