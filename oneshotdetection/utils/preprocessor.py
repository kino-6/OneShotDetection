import cv2
import numpy as np
import torch
from typing import Tuple, List


class Preprocessor:
    """
    Prepares images for model inference by resizing, normalizing, and converting to tensor format.
    """

    DEFAULT_COLOR: int = 0  # Default padding color for letterbox resizing

    def __init__(self, input_shape: Tuple[int, int] = (640, 640)):
        """
        Initializes the preprocessor.

        Args:
            input_shape (Tuple[int, int]): Target image size (height, width) for the model input.
        """
        self.input_shape = input_shape

    def letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes an image to the target shape using letterbox resizing.

        Letterbox resizing maintains the aspect ratio of the original image and pads the
        remaining area with a default color.

        Args:
            image (np.ndarray): Original image (H, W, C).

        Returns:
            Tuple[np.ndarray, float, Tuple[int, int]]:
                - Resized image with padding (H', W', C).
                - Scaling factor used for resizing.
                - Padding applied (left, top).
        """
        original_height, original_width = image.shape[:2]
        target_height, target_width = self.input_shape

        scale = min(target_width / original_width, target_height / original_height)
        new_width, new_height = int(original_width * scale), int(original_height * scale)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create a canvas with the default padding color
        canvas = np.full((target_height, target_width, 3), self.DEFAULT_COLOR, dtype=np.uint8)
        top, left = (target_height - new_height) // 2, (target_width - new_width) // 2

        # Place the resized image on the canvas
        canvas[top:top + new_height, left:left + new_width, :] = resized_image
        return canvas, scale, (left, top)

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[float, Tuple[int, int]]]:
        """
        Preprocesses an image for YOLOv5-style inference.

        The preprocessing includes resizing via letterbox, normalization, and converting
        to a tensor suitable for model input.

        Args:
            image (np.ndarray): Input image (H, W, C).

        Returns:
            Tuple[torch.Tensor, Tuple[int, int], Tuple[float, Tuple[int, int]]]:
                - Preprocessed image tensor (1, C, H, W).
                - Original image size (height, width).
                - Rescaling factor and padding details (scale, (left, top)).
        """
        # Get the original image size
        original_size = image.shape[:2]

        # Apply letterbox resizing
        processed_image, scale, padding = self.letterbox(image)

        # Convert BGR to RGB and normalize to [0, 1]
        processed_image = processed_image[:, :, ::-1].astype(np.float32) / 255.0

        # Rearrange dimensions from HWC to CHW
        processed_image = processed_image.transpose(2, 0, 1)

        # Convert to tensor
        tensor = torch.from_numpy(processed_image).unsqueeze(0)  # Add batch dimension

        return tensor, original_size, (scale, padding)

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Loads an image from a file path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Loaded image (H, W, C).

        Raises:
            FileNotFoundError: If the image file does not exist or cannot be read.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found or cannot be read: {image_path}")
        return image
