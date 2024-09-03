import tensorflow as tf
import numpy as np
from nltk.app.wordnet_app import explanation
from tensorflow.python.ops.numpy_ops import positive
# from tensorflow_probability.python.internal.backend.jax import negative

from tf_keras_vis.utils.scores import BinaryScore
from tf_keras_vis.saliency import Saliency
from alibi.explainers import IntegratedGradients
# TODO
from lime.lime_text import LimeTextExplainer
from utils import pad_inputs, WORD_TO_ID, indices2words
from config import MAX_SEQUENCE_LENGTH, LENGTH_EXPLANATION, DEFAULT_WORD_ID

def manual_saliency_map_embedding(model, input_tensor, target_class=0, magnitude=True):
    """
    Compute SmoothGrad saliency maps for models with the first layer as an embedding layer.

    Args:
    - model: The Keras model.
    - input_tensor: The input tensor of indices (e.g., word indices).
    - target_class: The class index for which to compute the gradients. If None, the most likely class is used.
    - magnitude: If True, take the magnitude of gradients. Otherwise, use signed gradients.

    Returns:
    - grads: The saliency map.
    """

    # Ensure input_indices is a TensorFlow tensor and has batch dimension
    if not isinstance(input_tensor, tf.Tensor):
        input_tensor = tf.convert_to_tensor(input_tensor)
    if len(input_tensor.shape) == 1:
        input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension
    input_tensor = tf.cast(input_tensor, tf.int32)

    # Extract the embedding layer
    embedding_layer = model.get_layer(index=0)

    # Get the embeddings by passing through the embedding layer
    embeddings = embedding_layer(input_tensor)
    # print(embeddings)

    # Compute embeddings for the given index_list
    with tf.GradientTape() as tape:
        # Watch the embedding layer output
        tape.watch(embeddings)
    
        # Pass embeddings through the rest of the model
        x = embeddings
        for layer in model.layers[1:]:
            x = layer(x)
        predictions = x
    
        # Compute the score for the target class
        # Predict and get the loss for the target class
        if target_class is None:
            target_class = tf.argmax(predictions[0]).numpy()
        loss = predictions[:, target_class]
    
    # Compute the gradients with respect to the embeddings
    grads = tape.gradient(loss, embeddings)
    
    # Apply absolute value to the gradients
    if magnitude:
        grads = tf.abs(grads)
    
    print("Saliency map calculated successfully.")
    return grads

def back_propagate_embedding(gradients, method="sum"):
    """
    pseudo back propagation of embedding layer

    Args:
    - input_indices: The input indices (e.g., word indices).
    - gradients: The computed gradients corresponding to each input index.
    - method: Aggregation method ("sum", "mean", or "norm") to compute scalar values from gradients.

    Returns:
    - scalar_gradients
    """
    # Aggregate gradients across embedding dimensions
    if method == "sum":
        scalar_gradients = tf.reduce_sum(gradients, axis=-1)
    elif method == "mean":
        scalar_gradients = tf.reduce_mean(gradients, axis=-1)
    elif method == "norm":
        scalar_gradients = tf.norm(gradients, axis=-1)
    elif method == "max":
        scalar_gradients = tf.reduce_max(gradients, axis=-1)
    else:
        raise ValueError("Unknown method: choose 'sum', 'mean', 'max',or 'norm'")
    scalar_gradients = scalar_gradients.numpy()
    return scalar_gradients

def top_k_attributions(input_indices, attributions, k=LENGTH_EXPLANATION):
    """
    rank_input_indices_by_gradient
    sort input indices based on the magnitude of gradient
    Args:
    - input_indices: The input indices (e.g., word indices).
    - attributions: The computed gradients corresponding to each input index.

    Returns:
    - sorted_input_indices: Indices sorted by magnitude of attributions (high to low).
    - sorted_attributions: Sorted gradient values corresponding to the ranked indices.
    - ranked_indices: Indices of the sorted gradients.
    """

    # Convert to numpy for sorting
    if not isinstance(input_indices, np.ndarray):
        input_indices = np.array(input_indices).squeeze()
    if not isinstance(attributions, np.ndarray):
        attributions = np.array(attributions).squeeze()

    #if magnitude:
    attributions_magnitude = np.abs(attributions)
    # Rank indices by gradient magnitude (descending order)
    ranked_indices = np.argsort(-attributions_magnitude)[0:k]

    # Sort input indices based on ranking
    sorted_input_indices = input_indices[ranked_indices]

    sorted_attributions = attributions[ranked_indices]

    return sorted_input_indices, sorted_attributions, ranked_indices

"""else:
    positive_attributions = np.clip(attributions, 0, None)
    negative_attributions = np.clip(attributions, None, 0)

    # Rank indices by gradient magnitude (descending order)
    ranked_indices_pos = np.argsort(-positive_attributions)[0:k//2]
    ranked_indices_neg = np.argsort(negative_attributions)[0:k//2]

    # Sort input indices based on ranking
    sorted_input_indices_pos = input_indices[ranked_indices_pos]
    sorted_input_indices_neg = input_indices[ranked_indices_neg]

    sorted_gradients_pos = attributions[ranked_indices_pos]
    sorted_gradients_neg = attributions[ranked_indices_neg]

# print(f"sorted word ", [(id, pre.id_to_word[id], grad) for id, grad in zip(sorted_input_indices, sorted_gradients)])

    return (sorted_input_indices_pos, sorted_input_indices_neg), (sorted_gradients_pos, sorted_gradients_neg)"""


def xai_embedding(model, input_list, xai_method, target_class):
    # convert different input into a TensorFlow tensor that matches input dimension
    if isinstance(input_list, tf.Tensor):
        input_tensor = input_list
    elif isinstance(input_list, list):
        if isinstance(input_list[0], int):
            input_list = np.array(input_list)
            input_list = np.expand_dims(input_list, 0)
        elif isinstance(input_list[0], list):
            input_list = np.array(input_list)
        input_list = pad_inputs(input_list)
        input_tensor = tf.convert_to_tensor(input_list)
    elif isinstance(input_list, np.ndarray):
        input_list = pad_inputs(input_list)
        input_tensor = tf.convert_to_tensor(input_list)

    if len(input_tensor.shape) == 1:
        input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension
    input_tensor = tf.cast(input_tensor, tf.int32)

    # Extract the embedding layer
    embedding_layer = model.get_layer(index=0)

    # Get the embeddings by passing through the embedding layer
    embeddings = embedding_layer(input_tensor)
    # print(embeddings.shape)

    # Create a new model that starts from the second layer onward
    submodel_input = tf.keras.Input(shape=embeddings.shape[1:])
    x = submodel_input
    for layer in  model.layers[1:]:
        x = layer(x)

    submodel_output = x
    submodel = tf.keras.Model(inputs=submodel_input, outputs=submodel_output)
    # submodel.summary()

    # Generate saliency map
    if xai_method == "VanillaSaliency":
        attributions = vanilla_saliency(submodel, embeddings, target_class)
        saliency_map = back_propagate_embedding(attributions, method="norm")
    elif xai_method == "SmoothGrad":
        attributions = smooth_grad(submodel, embeddings, target_class)
        saliency_map = back_propagate_embedding(attributions, method="norm")
    elif xai_method == "IntegratedGradients":
        attributions = integrated_gradients(submodel, embeddings, target_class)
        saliency_map = back_propagate_embedding(attributions, method="sum")
    else:
        raise ValueError("Unknown method: choose 'VanillaSaliency', 'SmoothGrad','IntegratedGradients'")
    saliency_map = np.reshape(saliency_map, (-1, MAX_SEQUENCE_LENGTH))
    print(f"Saliency map {xai_method} calculated successfully.")
    return saliency_map


def vanilla_saliency(model, X, target_class):
    # Create Saliency object.
    saliency = Saliency(model,
                        model_modifier=None,
                        clone=True)

    # Generate saliency map
    saliency_map = saliency(BinaryScore(target_class), X,
                            #gradient_modifier=None,
                            normalize_map=False, # normalize to (1., 0.).
                            keepdims=True)
    return saliency_map


def smooth_grad(model, X, target_class):
    # Create Saliency object.
    saliency = Saliency(model,
                        model_modifier=None,
                        clone=True)

    # Generate saliency map with smoothing that reduce noise by adding noise
    saliency_map = saliency(BinaryScore(target_class),
                            X,
                            smooth_samples=2,  # The number of calculating gradients iterations. 20
                            smooth_noise=0.20, # noise spread level.
                            #gradient_modifier=None,
                            normalize_map=False, # normalize to (1., 0.).
                            keepdims=True)
    return saliency_map


def integrated_gradients(model, X, target_class, steps=5): # 10
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    ig = IntegratedGradients(model,
                            n_steps=steps,
                            method="gausslegendre")

    # predictions = np.ones((X.shape[0])) * target_class
    # predictions = predictions.astype(int)
    explanation = ig.explain(X,
                            baselines=None,
                            target=target_class)
    attributions = np.array(explanation.attributions).squeeze()

    # print(attributions.shape)
    # assert attributions.shape ==
    # positive_attributions = np.clip(attributions, 0, None)
    # negative_attributions = np.clip(attributions, None, 0)

    return attributions # positive_attributions, negative_attributions


def lime_explainer(prediction_function, text, length_explanation=LENGTH_EXPLANATION):
    """
    Lime text explainer
    Input:
    - prediction_function: The prediction function to be explained
    - text: The text to be explained
    - length_explanation: The length of the explanation
    Output:
    - explanation: a list of explanation [(words, attributions),...]
    """
    explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
    # Generate explanation
    exp = explainer.explain_instance(text,
                                     prediction_function,
                                     num_features=length_explanation)
    return exp.as_list()
    # Display the explanation
    # exp.show_in_notebook(text=True)

def lime_batch_explainer(prediction_function, indices_list, label_list, length_explanation=LENGTH_EXPLANATION):
    """
    Lime text explainer
    Input:
    - prediction_function: The prediction function to be explained
    - text: The text to be explained
    - length_explanation: The length of the explanation
    Output:
    - explanation: a list of explanation [(words, attributions),...]
    """
    explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

    explanations = []
    # Generate explanation
    for indices in indices_list:
        text = indices2words(indices)
        exp = explainer.explain_instance(text,
                                         prediction_function,
                                         num_features=length_explanation)

        explanation = [(WORD_TO_ID.get(word, DEFAULT_WORD_ID["<unk>"]), attribution) for (word, attribution) in exp.as_list()]
        explanations.append(explanation)
    return explanations


if __name__ == "__main__":
    from population import load_imdb_test
    from predictor import Predictor
    from utils import pad_inputs
    X_test, y_test = load_imdb_test(pop_size = 10)

    print(X_test.shape, y_test.shape)

    predictor = Predictor()
    # prediction = predictor.predict(X_test)
    # print(prediction)
    #
    # explanation = xai_embedding(predictor.model, X_test[0], "IntegratedGradients")
    # print(explanation.shape)
    # explanations = xai_embedding(predictor.model, X_test, "IntegratedGradients")
    # print(explanations.shape)


    text = indices2words(X_test[0])
    texts = [indices2words(X_test[0]), indices2words(X_test[1])]
    explanations_lime = lime_batch_explainer(predictor.predict_texts_xai, X_test[0:2])
    print(explanations_lime)