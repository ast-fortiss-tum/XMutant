import numpy as np
import torch
from config import VIT_MODEL_CONFIGS
import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from transformer_model.transformer_package.models.transformer import ViT
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class Predictor:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViT(
            image_size=VIT_MODEL_CONFIGS["image_size"],
            channel_size=VIT_MODEL_CONFIGS["channel_size"],
            patch_size=VIT_MODEL_CONFIGS["patch_size"],
            embed_size=VIT_MODEL_CONFIGS["embed_size"],
            num_heads=VIT_MODEL_CONFIGS["num_heads"],
            classes=VIT_MODEL_CONFIGS["classes"],
            num_layers=VIT_MODEL_CONFIGS["num_layers"],
            hidden_size=VIT_MODEL_CONFIGS["hidden_size"],
            dropout=VIT_MODEL_CONFIGS["dropout"],
        ).to(self.device)

        # Load the checkpoint
        checkpoint = torch.load(VIT_MODEL_CONFIGS["checkpoint_path"], map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint with epoch {checkpoint['epoch']} and validation accuracy {checkpoint['val_accuracy']:.4f}.")

    def predict(self, images):
        """
        Predicts classes and confidences for a single image or group of vit_model in NumPy format.

        Args:
            images (numpy.ndarray): Input image or vit_model in NumPy array format.
                                    Shape should be [H, W, C] for single image or [N, H, W, C] for multiple vit_model.
            transform (callable, optional): Transform to preprocess the input image(s), if necessary.

        Returns:
            predictions (list): Predicted class indices for each input.
            confidences (list): Confidence scores for the predicted classes.
        """
        self.model.eval()  # Set the model to evaluation mode

        # Ensure vit_model are in batch format: [N, H, W]
        if len(images.shape) == 2:  # Single image [H, W]
            images = np.expand_dims(images, axis=0)  # Convert to [1, H, W]

        # Convert vit_model to PyTorch tensor with transform
        mean, std = (0.5,), (0.5,)
        transform = transforms.Compose([
            transforms.Pad(padding=2),  # Pad 28x28 MNIST vit_model to 32x32
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Apply transform to each image and stack into a batch tensor
        processed_images = torch.stack([transform(Image.fromarray((image.squeeze()).astype(np.uint8)))
                                        for image in images])
        processed_images = processed_images.to(self.device, dtype=torch.float32)

        predictions = []
        confidences = []

        with torch.no_grad():  # Disable gradient calculation
            # Get model outputs
            outputs, attention_weights = self.model(processed_images)  # Shape: [batch_size, num_classes]

            # Compute probabilities using softmax
            probs = F.softmax(outputs, dim=-1)  # Shape: [batch_size, num_classes]

            # Get predicted classes and their confidence scores
            pred_classes = probs.argmax(dim=-1)  # Shape: [batch_size]
            pred_confidences = probs.max(dim=-1).values  # Shape: [batch_size]

            predictions.extend(pred_classes.tolist())
            confidences.extend(pred_confidences.tolist())

            # Compute attention rollout
            rollout = attention_rollout(attention_weights, num_layers=self.model.num_layers, device=self.device)
            # Normalize for visualization if necessary
            rollout = rollout.mean(dim=1)  # You can average over heads if needed
            rollout = rollout.squeeze().cpu().numpy()  # Convert to numpy for visualization

        return predictions, confidences

if __name__ == "__main__":
    from population import load_mnist_test

    pop_size = 10
    digit = 5
    x_test, y_test = load_mnist_test(pop_size, digit)
    x_test = np.expand_dims(x_test, -1)
    predictor = Predictor()
    predictions, confidences = predictor.predict(x_test)

    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        print(f"Image {i+1}: Predicted Class = {pred}, Confidence = {conf:.4f}, label = {y_test[i]}")

