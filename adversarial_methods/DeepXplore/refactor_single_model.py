from keras.layers import Input
from keras.utils import to_categorical
from utils import *
import numpy as np
import random
import imageio
from Model1 import Model1

# Parameters (replace these with your desired values or use argparse as needed)
transformation = 'light'  # Options: 'light', 'occl', 'blackout'
weight_diff = 1
weight_nc = 0.1
weight_vae = 0
step = 0.1
seeds = 50
grad_iterations = 2500
threshold = 0.0
target_model = 0

# Prepare data (MNIST dataset example)
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255.0

# Select specific class for perturbation
CLASS = 5
idxs = np.argwhere(y_test == CLASS)
starting_seeds = x_test[idxs]
starting_seeds = starting_seeds[:seeds]  # Limit to desired number of seeds
starting_seeds = starting_seeds.reshape(starting_seeds.shape[0], 28, 28, 1)

# Model setup
input_tensor = Input(shape=input_shape)
model = Model1(input_tensor=input_tensor)
model_layer_dict = init_coverage_tables(model)

# Perturbation loop
for seed_idx, current_seed in enumerate(starting_seeds):
    print(f"Processing seed {seed_idx + 1}/{len(starting_seeds)}")
    gen_img = np.expand_dims(current_seed, axis=0)
    orig_label = np.argmax(model.predict(gen_img)[0])

    # Construct loss function
    loss1 = -weight_diff * K.mean(model.get_layer('before_softmax').output[..., orig_label])
    loss1_neuron = K.mean(model.get_layer('layer_name_to_cover').output[..., index1])
    final_loss = loss1 + weight_nc * loss1_neuron

    # Compute gradients
    grads = normalize(K.gradients(final_loss, input_tensor)[0])
    iterate = K.function([input_tensor], [loss1, loss1_neuron, grads])

    # Gradient ascent iterations
    for iteration in range(grad_iterations):
        loss_value1, loss_neuron, grads_value = iterate([gen_img])

        # Apply constraints (transformation-specific)
        if transformation == 'light':
            grads_value = constraint_light(grads_value)
        elif transformation == 'occl':
            grads_value = constraint_occl(grads_value, start_point=(0, 0), occlusion_size=(10, 10))
        elif transformation == 'blackout':
            grads_value = constraint_black(grads_value)

        # Update the generated image
        gen_img += grads_value * step
        gen_img = np.clip(gen_img, 0, 1)

        # Check for prediction changes
        new_prediction = np.argmax(model.predict(gen_img)[0])
        if new_prediction != orig_label:
            print(f"Misbehavior detected at iteration {iteration}")
            gen_img_deprocessed = deprocess_image(gen_img)
            imageio.imwrite(f'output/seed_{seed_idx}_iteration_{iteration}.png', gen_img_deprocessed)
            break
