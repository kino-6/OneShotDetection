import torch
import torch.nn.functional as F
from typing import Optional
import math


class ImageProcessing:
    """
    Utility class for image processing tasks such as resizing and padding.
    """

    @staticmethod
    def scale_image(
        img: torch.Tensor,
        scale_ratio: float = 1.0,
        keep_shape: bool = False,
        grid_size: int = 32,
        padding_value: float = 0.447
    ) -> torch.Tensor:
        """
        Scales the input image tensor by a specified ratio while maintaining grid constraints.

        Args:
            img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            scale_ratio (float): Scaling ratio for height and width. Default is 1.0 (no scaling).
            keep_shape (bool): If True, output image size matches grid size multiple. Default is False.
            grid_size (int): Grid size for padding or cropping. Default is 32.
            padding_value (float): Padding value for the resized image. Default is 0.447.

        Returns:
            torch.Tensor: Resized and optionally padded image tensor.
        """
        # Return early if no scaling is required
        if scale_ratio == 1.0:
            return img

        # Compute the new height and width
        original_height, original_width = img.shape[2:]
        new_height, new_width = int(original_height * scale_ratio), int(original_width * scale_ratio)

        # Resize the image
        resized_img = F.interpolate(img, size=(new_height, new_width), mode="bilinear", align_corners=False)

        # If keep_shape is True, pad or crop the image to align with the grid size
        if not keep_shape:
            padded_height = math.ceil(new_height / grid_size) * grid_size
            padded_width = math.ceil(new_width / grid_size) * grid_size
            padding = [0, padded_width - new_width, 0, padded_height - new_height]
            resized_img = F.pad(resized_img, padding, value=padding_value)

        return resized_img
