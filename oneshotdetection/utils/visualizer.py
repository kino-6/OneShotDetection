import os
from typing import List, Tuple, Dict
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """
    A class for visualizing object detection results with consistent class colors.
    """

    def __init__(self, output_dir: str = "runs", num_classes: int = 80):
        """
        Initialize the Visualizer.

        Args:
            output_dir (str): Directory to save visualized images.
            num_classes (int): Total number of classes for which colors will be predefined.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_palette = self._generate_tab20_palette(num_classes)

    @staticmethod
    def _generate_tab20_palette(num_classes: int) -> List[Tuple[int, int, int]]:
        """
        Generate a visually distinct color palette using Matplotlib's 'tab20'.

        Args:
            num_classes (int): Number of classes to generate colors for.

        Returns:
            List[Tuple[int, int, int]]: List of BGR color tuples.
        """
        cmap = plt.cm.get_cmap("tab20", num_classes)  # Use 'tab20' colormap
        colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_classes)]
        return [(b, g, r) for r, g, b in colors]  # Convert RGB to BGR for OpenCV

    def _generate_run_dir(self) -> Path:
        """
        Generate a unique directory for each run.

        Returns:
            Path: Directory path for the current run.
        """
        run_id = 0
        while True:
            run_dir = self.output_dir / f"runs_{run_id:05d}"
            if not run_dir.exists():
                run_dir.mkdir(parents=True)
                return run_dir
            run_id += 1

    def visualize_and_save(
        self,
        image: np.ndarray,
        detections: List[Dict[str, object]],
        output_name: str,
    ) -> str:
        """
        Visualize detections on the image and save the result.

        Args:
            image (np.ndarray): Input image in BGR format.
            detections (List[Dict[str, object]]): List of detection results with "box", "confidence", "class_name".
            output_name (str): Name of the output file.

        Returns:
            str: Path to the saved image.
        """
        run_dir = self._generate_run_dir()

        # Draw detections on the image
        for det in detections:
            box = det["box"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            class_id = det.get("class_id", 0)  # Default to 0 if not provided
            color = self.color_palette[class_id % len(self.color_palette)]
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Add label and confidence
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_y1 = max(y1 - label_size[1] - 10, 0)
            label_x2 = min(x1 + label_size[0], image.shape[1])
            cv2.rectangle(image, (x1, label_y1), (label_x2, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save the visualized image
        output_path = run_dir / output_name
        cv2.imwrite(str(output_path), image)
        print(f"Visualization saved to: {output_path}")
        return str(output_path)
