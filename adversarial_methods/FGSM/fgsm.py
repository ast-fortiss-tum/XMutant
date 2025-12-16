import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model

# from xai import gradient_of_x
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.activations import softmax


def remove_last_softmax_activation(model):
    if isinstance(model.layers[-1], Dense) and getattr(model.layers[-1], "activation") == softmax:
        # Modify the last layer to have a linear activation
        model_clone = tf.keras.models.clone_model(model)
        model_clone.set_weights(model.get_weights())
        model_clone.layers[-1].activation = tf.keras.activations.linear
        model = Model(inputs=model_clone.inputs, outputs=model_clone.layers[-1].output)
    return model


def gradient_of_x(x, y, model, before_softmax=False):
    # Check if the last layer is a Dense layer with softmax activation
    if before_softmax:
        model = remove_last_softmax_activation(model)

    # Convert the numpy arrays to TensorFlow tensors
    input_data = tf.convert_to_tensor(x, dtype=tf.float32)
    true_labels = tf.convert_to_tensor(y, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_data)  # Explicitly watch the input tensor

        # Now directly feeding `input_data` to the model, so TensorFlow automatically tracks operations
        predictions = model(input_data, training=False)

        # Compute the categorical cross-entropy loss
        loss = tf.keras.losses.categorical_crossentropy(true_labels, predictions)

    # Compute the gradient of the loss with respect to the input
    return tape.gradient(loss, input_data)


class FGSM:

    def __init__(self, model):
        self.model = model

    def generate_adversarial_image(self, image, image_label, step=0.5):
        """

        Args:
            image: np.array, of input shape of the self.model
            image_label: onehot vector
            step:

        Returns:

        """
        image = np.reshape(image, self.model.input_shape[1:])
        x = np.array([image])
        y = np.array([image_label])
        g = gradient_of_x(x, y, self.model)
        g_npy = np.squeeze(g.numpy())
        g_sign = np.reshape(tf.sign(g_npy), self.model.input_shape[1:])

        image_adv = np.copy(image)  # _lava means latent variants
        image_adv += step * g_sign
        return image_adv
