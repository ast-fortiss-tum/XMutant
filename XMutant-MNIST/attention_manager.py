import numpy as np
from alibi.explainers import IntegratedGradients
from config import ATTENTION
from predictor import Predictor
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# import keras


class AttentionManager:
    def __init__(self, num, attention_method=ATTENTION):
        self.digit = num
        self.score = CategoricalScore(self.digit)
        self.replace2linear = ReplaceToLinear()
        self.model = Predictor.model
        self.attention_method = attention_method

    def compute_attention_maps(
        self, images
    ):  # vit_model should have the shape: (x, 28, 28) where x>=1
        X = self.get_input(images)

        switch = {
            "VanillaSaliency": self.vanilla_saliency,
            "SmoothGrad": self.smooth_grad,
            "GradCAM": self.grad_cam,
            "GradCAM++": self.grad_cam_pp,
            "ScoreCAM": self.score_cam,
            "Faster-ScoreCAM": self.faster_score_cam,
            "IntegratedGradients": self.integrated_gradients,
        }
        attention_maps = switch.get(self.attention_method)(X)
        return attention_maps

    def vanilla_saliency(self, X):
        # Create Saliency object.
        saliency = Saliency(self.model, model_modifier=self.replace2linear, clone=True)

        # Generate saliency map
        saliency_map = saliency(self.score, X)
        return saliency_map

    def smooth_grad(self, X):
        # Create Saliency object.
        saliency = Saliency(self.model, model_modifier=self.replace2linear, clone=True)

        # Generate saliency map with smoothing that reduce noise by adding noise
        saliency_map = saliency(
            self.score,
            X,
            smooth_samples=2,  # The number of calculating gradients iterations. 20
            smooth_noise=0.20,
        )  # noise spread level.
        return saliency_map

    def grad_cam(self, X):
        # Create Gradcam object
        gradcam = Gradcam(self.model, model_modifier=self.replace2linear, clone=True)

        # Generate heatmap with GradCAM
        cam = gradcam(self.score, X, penultimate_layer=-1)
        return cam

    def grad_cam_pp(self, X):
        # Create GradCAM++ object
        gradcam = GradcamPlusPlus(self.model, model_modifier=self.replace2linear, clone=True)

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
        scorecam = Scorecam(self.model, model_modifier=self.replace2linear)

        # Generate heatmap with Faster-ScoreCAM
        cam = scorecam(self.score, X, penultimate_layer=-1, max_N=2)  # 10
        return cam

    def integrated_gradients(self, X, steps=5):  # 10
        ig = IntegratedGradients(self.model, n_steps=steps, method="gausslegendre")
        # predictions = self.model(X).numpy().argmax(axis=1)
        predictions = np.ones((X.shape[0])) * self.digit
        predictions = predictions.astype(int)
        explanation = ig.explain(X, baselines=None, target=predictions)
        attributions = explanation.attributions[0]
        # remove single-dimensional shape of the array.
        # attributions = attributions.squeeze()
        attributions = np.reshape(attributions, (-1, 28, 28))
        # only focus on positive part
        # attributions = attributions.clip(0, 1)
        attributions = np.abs(attributions)
        normalized_attributions = np.zeros(shape=attributions.shape)

        # Normalization
        # Todo in numpy divide by zero returns nan...
        for i in range(attributions.shape[0]):
            try:
                # print(f"attention map difference {np.max(attributions[i]) - np.min(attributions[i])}")
                normalized_attributions[i] = (attributions[i] - np.min(attributions[i])) / (
                    np.max(attributions[i]) - np.min(attributions[i])
                )
            except ZeroDivisionError:
                print("Error: Cannot divide by zero")
                return
        # print(normalized_attributions.shape)
        return normalized_attributions

    def get_input(self, x_test):
        X = np.reshape(x_test, (-1, 28, 28, 1))
        X = X.astype("float32")
        # X /= 255.0
        return X

    """def input_reshape_and_normalize_images(self, x):
        # shape numpy vectors
        if keras.backend.image_data_format() == 'channels_first':
            x_reshape = x.reshape(x.shape[0], 1, 28, 28)
        else:
            x_reshape = x.reshape(x.shape[0], 28, 28, 1)
        x_reshape = x_reshape.astype('float32')
        x_reshape /= 255.0
        return x_reshape"""


if __name__ == "__main__":
    from population import load_mnist_test

    popsize = 3
    number = 3
    x_test, y_test = load_mnist_test(popsize, number)

    print(x_test.shape)
    model = Predictor.model

    for i in range(popsize):
        # print(x_test[i])
        input = np.reshape(x_test[i], (1, 28, 28))
        print(f"Prediction {model.predict(input)}")

    am = AttentionManager(num=number, attention_method="IntegratedGradients")
    am.compute_attention_maps(x_test)

    # pop.evaluate_population(0)
