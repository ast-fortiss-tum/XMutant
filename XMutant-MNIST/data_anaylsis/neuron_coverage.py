import tensorflow as tf
import numpy as np

import sys
import os
import gzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# os.chdir("..")
print(os.getcwd())

from config import MODEL, num_classes



def load_mnist_test_(popsize, number):
    file_test_x = '../original_dataset/t10k-images-idx3-ubyte.gz'
    file_test_y = '../original_dataset/t10k-labels-idx1-ubyte.gz'

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

# Function to get activations from a layer
def get_layer_outputs(model, inputs):
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    return activation_model.predict(inputs)

def calculate_neuron_coverage(layer_activations, threshold=0.5):
    activated_neurons = (layer_activations > threshold).sum()
    total_neurons = layer_activations.size
    return activated_neurons / total_neurons

def model_neuron_coverage(model, inputs, threshold=0.5):
    activations = get_layer_outputs(model, inputs)
    total_coverage = 0.0
    for layer_activation in activations:
        total_coverage += calculate_neuron_coverage(layer_activation, threshold)
    return total_coverage / len(activations)

if __name__ == "__main__":
    pop_size = 10
    digit = 5
    test_data, test_labels = load_mnist_test_(pop_size, 5)

    # Load your TensorFlow model
    model = tf.keras.models.load_model(os.path.join(".." ,MODEL))

    # Example for a single layer
    layer_activations = get_layer_outputs(model, test_data)[0]  # Assuming test_data is your input
    neuron_coverage = calculate_neuron_coverage(layer_activations)
    print(f"Neuron Coverage: {neuron_coverage:.2f}")

    # Calculate overall neuron coverage
    coverage = model_neuron_coverage(model, test_data)
    print(f"Model Neuron Coverage: {coverage:.2f}")

