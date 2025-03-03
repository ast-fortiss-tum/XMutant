import numpy as np
import torch
from matplotlib.pyplot import colorbar

from config import VIT_MODEL_CONFIGS
import os
import sys

from vit_model.vit_model import VisionTransformer, img_to_patch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from config import VIT_MODEL_CONFIGS, BATCH_SIZE

class Predictor:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionTransformer(
                          embed_dim=VIT_MODEL_CONFIGS["embed_dim"],
                          hidden_dim=VIT_MODEL_CONFIGS["hidden_dim"],
                          num_heads=VIT_MODEL_CONFIGS["num_heads"],
                          num_layers=VIT_MODEL_CONFIGS["num_layers"],
                          patch_size=VIT_MODEL_CONFIGS["patch_size"],
                          num_channels=VIT_MODEL_CONFIGS["num_channels"],
                          num_patches=VIT_MODEL_CONFIGS["num_patches"],
                          num_classes=VIT_MODEL_CONFIGS["num_classes"],
                          dropout=VIT_MODEL_CONFIGS["dropout"],
            ).to(self.device)

        # Load the checkpoint
        checkpoint = torch.load(VIT_MODEL_CONFIGS["checkpoint_path"], map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set the model to evaluation mode
        print(f"Loaded checkpoint with epoch {checkpoint['epoch']} and validation accuracy {checkpoint['val_accuracy']:.4f}.")

        # Convert vit_model to PyTorch tensor with transform
        if VIT_MODEL_CONFIGS["normalization"]:
            mean, std = (0.5,), (0.5,)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])


    def predict(self, images: np.ndarray, batch_size=BATCH_SIZE):
        """
        Predicts classes and confidences for a single image or group of vit_model in NumPy format.

        Args:
            images (numpy.ndarray): Input image or vit_model in NumPy array format.
                                    Shape should be [H, W, C] for single image or [N, H, W, C] for multiple vit_model.
            batch_size (int): Size of the mini-batches for prediction. Default is 64.

        Returns:
            predictions (list): Predicted class indices for each input.
            confidences (list): Confidence scores for the predicted classes.
        """

        # Ensure vit_model are in batch format: [N, H, W]
        if len(images.shape) == 2:  # Single image [H, W]
            images = np.expand_dims(images, axis=0)  # Convert to [1, H, W]

        predictions = []
        confidences = []

        # Iterate over the images in mini-batches
        for i in range(0, len(images), batch_size):
            mini_batch = images[i:i + batch_size]  # Get the current mini-batch

            # Transform and stack the mini-batch
            processed_images = torch.stack([
                self.transform(Image.fromarray((image.squeeze())))
                for image in mini_batch
            ])  # Shape: [mini_batch_size, channels, height, width]

            # Move to the appropriate device
            processed_images = processed_images.to(self.device, dtype=torch.float32)

            with torch.no_grad():  # Disable gradient calculation
                # Forward pass through the model
                outputs = self.model(processed_images)  # Shape: [mini_batch_size, num_classes]

                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=-1)  # Shape: [mini_batch_size, num_classes]

                # Get predicted class indices and confidence scores
                pred_classes = probs.argmax(dim=-1)  # Shape: [mini_batch_size]
                pred_confidences = probs.max(dim=-1).values  # Shape: [mini_batch_size]

                # Extend the results to the final lists
                predictions.extend(pred_classes.tolist())
                confidences.extend(pred_confidences.tolist())

        del processed_images
        return predictions, confidences

    def attention_rollout(self, images, batch_size=BATCH_SIZE):
        # Ensure vit_model are in batch format: [N, H, W]
        if len(images.shape) == 2:  # Single image [H, W]
            images = np.expand_dims(images, axis=0)  # Convert to [1, H, W]

        attn_heatmaps_resized = None

        # Iterate over the images in mini-batches
        for i in range(0, len(images), batch_size):
            mini_batch = images[i:i + batch_size]  # Get the current mini-batch
            bs = mini_batch.shape[0]
            # Transform and stack the mini-batch
            processed_images = torch.stack([
                self.transform(Image.fromarray((image.squeeze()).astype(np.uint8)))
                for image in mini_batch
            ]).to(self.device, dtype=torch.float32)  # Shape: [mini_batch_size, channels, height, width]

            with (torch.no_grad()):  # Disable gradient calculation
                # convert the vit_model sample into patches
                patches = img_to_patch(processed_images, patch_size=VIT_MODEL_CONFIGS["patch_size"])  # [B, num_patches,  patch_size²] [10, 49, 16]
                # run the patches through the input layer to get a tensor of size embed_dim
                patches = self.model.input_layer(patches.float()) # [B, num_patches, embed_dim] [10, 49, 256]
                # attach the class token and add the position embedding
                cls_token_expanded = self.model.cls_token.repeat(bs, 1, 1)  # Shape: [10, 1, 256]
                transformer_input = torch.cat((cls_token_expanded, patches), dim=1) + self.model.pos_embedding # [1, 50, 256]

                # run the embedded vit_model image through the first attention block and squeeze the
                # batch dimension because we're only using one vit_model image
                transformer_input_expanded = self.model.transformer[0].linear[0](transformer_input).squeeze(0) # [10, 50, 768]
                # reshape the output of the first attention block to be of size (bs, num_patches+1, 3, num_heads, -1) [10, 50, 3, 8, 32]
                qkv = transformer_input_expanded.reshape(bs, # batch_size
                                                         VIT_MODEL_CONFIGS['num_patches']+1, #  Sequence length (patches + class token)
                                                         3, # Query, Key, Value
                                                         VIT_MODEL_CONFIGS['num_heads'],
                                                         -1)  # Head size

                # pull the query matrix and permute the dimensions to be (8 heads, 17 patches, 32 channels)
                # do the same for the key matrix
                q = qkv[:, :, 0].permute(0, 2, 1, 3) # q = qkv[:, 0].permute(1, 0, 2) # [10, 8, 50, 32]
                k = qkv[:, :, 1].permute(0, 2, 1, 3) # k = qkv[:, 1].permute(1, 0, 2) # [10, 8, 50, 32]
                kT = k.permute(0, 1, 3, 2) # kT = k.permute(0, 2, 1) # [10, 8, 32, 50]
                # The result of multiplying q @ kT is a squared matrix 17 by 17 showing how each
                # patch is "paying attention" to every other patch
                attention_matrix = torch.matmul(q, kT) # q @ kT  # [10,8, 50, 50]

                # Average the attention weights across all heads by taking the mean along
                # the first dimension
                attention_matrix_mean = torch.mean(attention_matrix, dim=1) # [bs, 50, 50]
                # print("attention matrix mean: ", attention_matrix_mean.shape)

                # To account for residual connections, we add an identity matrix to the attention matrix and re-normalize the weights.
                # Please refer to the attention rollout paper: https://arxiv.org/abs/2005.00928
                identity_matrix = torch.eye(attention_matrix_mean.size(1)).to(self.device) # [50, 50]
                residual_att = attention_matrix_mean + identity_matrix # [bs, 50, 50]
                # print("augmented attention matrix: ", aug_att_mat.shape)
                aug_att_mat = residual_att / residual_att.sum(dim=-1, keepdim=True)# .unsqueeze(-1) # [bs, 50, 50]
                # print("normalized augmented attention matrix: ", aug_att_mat.shape)


                for i in range(bs):
                    # Skip the [CLS] token and reshape attention to patch grid
                    attn_heatmap = aug_att_mat[i, 0, 1:].reshape(
                        (int(VIT_MODEL_CONFIGS["image_size"] / VIT_MODEL_CONFIGS["patch_size"]),
                         int(VIT_MODEL_CONFIGS["image_size"] / VIT_MODEL_CONFIGS["patch_size"]))
                    )  # Shape: [patch_grid, patch_grid]

                    # Normalize the attention heatmap
                    attn_heatmap = (attn_heatmap - attn_heatmap.min()) / (attn_heatmap.max() - attn_heatmap.min())

                    # Resize attention heatmap to original image size using bilinear interpolation
                    attn_heatmap_resized = F.interpolate(
                        attn_heatmap.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                        size=[VIT_MODEL_CONFIGS["image_size"], VIT_MODEL_CONFIGS["image_size"]],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().view(VIT_MODEL_CONFIGS["image_size"], VIT_MODEL_CONFIGS["image_size"], 1)  # Shape: [image_size, image_size, 1]

                    attn_heatmap_resized = attn_heatmap_resized.detach().cpu().numpy()

                    # Append resized heatmap to the list
                    if attn_heatmaps_resized is None:  # Initialize the NumPy array for the first time
                        attn_heatmaps_resized = attn_heatmap_resized[np.newaxis, ...]  # Add a new axis for batch
                    else:
                        attn_heatmaps_resized = np.append(attn_heatmaps_resized, attn_heatmap_resized[np.newaxis, ...], axis=0)

        return attn_heatmaps_resized

    def single_attention_rollout(self, image):
            # Transform and stack the mini-batch
            # img_tensor = torch.stack([
            #     self.transform(Image.fromarray((image.squeeze()).astype(np.uint8)))
            #         ]).to(self.device, dtype=torch.float32)  # Shape: [mini_batch_size, channels, height, width]
            # numpy to tensor
            image = np.expand_dims(image, 0)
            image = np.expand_dims(image, 0)
            img_tensor = torch.from_numpy(image)

            # with (torch.no_grad()):  # Disable gradient calculation
            # convert the vit_model sample into patches
            patches = img_to_patch(img_tensor, patch_size=VIT_MODEL_CONFIGS["patch_size"])  # [B, num_patches,  patch_size²] [10, 49, 16]
            # run the patches through the input layer to get a tensor of size embed_dim
            patches = self.model.input_layer(patches.float()) # [B, num_patches, embed_dim] [10, 49, 256]
            # attach the class token and add the position embedding
            # cls_token_expanded = self.model.cls_token.expand(patches.shape[0], -1, -1) # [1, 1, 256]
            transformer_input = torch.cat((self.model.cls_token, patches), dim=1) + self.model.pos_embedding # [1, 50, 256]

            # run the embedded vit_model image through the first attention block and squeeze the
            # batch dimension because we're only using one vit_model image
            transformer_input_expanded = self.model.transformer[0].linear[0](transformer_input).squeeze(0) # [10, 50, 768]
            # reshape the output of the first attention block to be of size (bs, num_patches+1, 3, num_heads, -1) [10, 50, 3, 8, 32]
            qkv = transformer_input_expanded.reshape(VIT_MODEL_CONFIGS['num_patches']+1, #  Sequence length (patches + class token)
                                                     3, # Query, Key, Value
                                                     VIT_MODEL_CONFIGS['num_heads'],
                                                     -1)  # Head size

            # pull the query matrix and permute the dimensions to be (8 heads, 17 patches, 32 channels)
            # do the same for the key matrix
            q = qkv[:, 0].permute(1, 0, 2)
            k = qkv[:, 1].permute(1, 0, 2)
            print("q shape: ", q.shape)
            print("k shape: ", k.shape)
            kT = k.permute(0, 2, 1)
            # The result of multiplying q @ kT is a squared matrix 17 by 17 showing how each
            # patch is "paying attention" to every other patch
            attention_matrix = q @ kT
            print("attention matrix: ", attention_matrix.shape)

            attention_matrix_mean = torch.mean(attention_matrix, dim=0)
            print("attention matrix mean: ", attention_matrix_mean.shape)

            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            # Please refer to the attention rollout paper: https://arxiv.org/abs/2005.00928
            residual_att = torch.eye(attention_matrix_mean.size(1)).to(self.device)
            aug_att_mat = attention_matrix_mean + residual_att
            print("augmented attention matrix: ", aug_att_mat.shape)
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
            print("normalized augmented attention matrix: ", aug_att_mat.shape)


            # Skip the [CLS] token and reshape attention to patch grid
            attn_heatmap = aug_att_mat[ 0, 1:].reshape(
                (int(VIT_MODEL_CONFIGS["image_size"] / VIT_MODEL_CONFIGS["patch_size"]),
                 int(VIT_MODEL_CONFIGS["image_size"] / VIT_MODEL_CONFIGS["patch_size"]))
            )  # Shape: [patch_grid, patch_grid]

            # Normalize the attention heatmap
            # attn_heatmap = (attn_heatmap - attn_heatmap.min()) / (attn_heatmap.max() - attn_heatmap.min())

            # Resize attention heatmap to original image size using bilinear interpolation
            attn_heatmap_resized = F.interpolate(
                attn_heatmap.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                size=[VIT_MODEL_CONFIGS["image_size"], VIT_MODEL_CONFIGS["image_size"]],
                mode='bilinear',
                #align_corners=False
            ).squeeze().view(VIT_MODEL_CONFIGS["image_size"], VIT_MODEL_CONFIGS["image_size"], 1)  # Shape: [image_size, image_size, 1]

            attn_heatmap_resized = attn_heatmap_resized.detach().cpu().numpy()

            return attn_heatmap_resized

if __name__ == "__main__":
    from population import load_mnist_test

    pop_size = 200
    for digit in range(10):
        x_test, y_test = load_mnist_test(pop_size, digit)
        # x_test = np.expand_dims(x_test, -1)
        predictor = Predictor()
        predictions, confidences = predictor.predict(x_test/255)

        # correctness of prediction
        print(f"Digit  {digit}, Accuracy: , {np.mean(predictions == y_test)}")


    # attn_heatmaps = predictor.attention_rollout(x_test)
    #
    # # Plot the attention heatmaps
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')
    # fig, axs= plt.subplots(pop_size, 3, figsize=(12, 60))
    # for i in range(pop_size):
    #     axs[i, 0].imshow(x_test[i], cmap='gray')
    #     #axs[i, 1].imshow(attn_heatmaps[i], cmap='jet')
    #     fig.colorbar(axs[i, 1].imshow(attn_heatmaps[i], cmap='jet'), ax=axs[i, 1])
    #
    #     # overlay the heatmap on the image
    #     axs[i, 2].imshow(x_test[i], cmap='gray')
    #     axs[i, 2].imshow(attn_heatmaps[i], cmap='jet', alpha=0.5)
    #
    # plt.show()

    # i = 5
    # attn_heatmap = predictor.single_attention_rollout(x_test[i])
    # # Plot the attention heatmaps
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')
    # fig, axs= plt.subplots(1, 3, figsize=(12, 60))
    #
    # axs[0].imshow(x_test[i], cmap='gray')
    # #axs[i, 1].imshow(attn_heatmaps[i], cmap='jet')
    # fig.colorbar(axs[1].imshow(attn_heatmap), ax=axs[1])
    #
    # # overlay the heatmap on the image
    # axs[2].imshow(x_test[i], cmap='gray')
    # axs[2].imshow(attn_heatmap, cmap='jet', alpha=0.5)
    # plt.show()
