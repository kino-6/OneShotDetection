import torch
from typing import List, Tuple, Dict
import torchvision.ops


class PostProcessor:
    """
    PostProcessor class for applying Non-Maximum Suppression (NMS) and rescaling detections.
    """

    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        class_names: List[str] = None
    ):
        """
        Initialize the PostProcessor.

        Args:
            confidence_threshold (float): Minimum confidence for filtering detections.
            iou_threshold (float): Intersection over Union (IoU) threshold for NMS.
            max_det (int): Maximum number of detections per image.
            class_names (List[str]): List of class names. Defaults to COCO_CLASSES.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.class_names = class_names or self.COCO_CLASSES

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert bounding boxes from [x, y, w, h] to [x1, y1, x2, y2] format.

        Args:
            boxes (torch.Tensor): Bounding boxes in [x, y, w, h] format.

        Returns:
            torch.Tensor: Bounding boxes in [x1, y1, x2, y2] format.
        """
        x, y, w, h = boxes.T
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return torch.stack((x1, y1, x2, y2), dim=1)

    def apply_nms(self, predictions: torch.Tensor) -> List[List[Dict[str, object]]]:
        """
        Apply Non-Maximum Suppression (NMS) on model predictions.

        Args:
            predictions (torch.Tensor): Model predictions of shape (batch_size, num_boxes, num_classes + 5).
                Each prediction includes [x, y, w, h, objectness, class_scores...].

        Returns:
            List[List[Dict[str, object]]]: A list of detections per image. Each detection contains:
                - "box": [x1, y1, x2, y2] (coordinates)
                - "confidence": float (confidence score)
                - "class_id": int (class index)
                - "class_name": str (class name)
        """
        batch_size = predictions.shape[0]
        outputs = []  # Store detections for each image

        for image_idx in range(batch_size):
            image_preds = predictions[image_idx]
            objectness_scores = image_preds[:, 4]
            objectness_mask = objectness_scores >= self.confidence_threshold
            image_preds = image_preds[objectness_mask]

            if image_preds.shape[0] == 0:
                outputs.append([])  # No detections for this image
                continue

            # Compute overall confidence
            class_probs = image_preds[:, 5:]
            class_scores, class_ids = torch.max(class_probs, dim=1)
            scores = objectness_scores[objectness_mask] * class_scores

            # Filter by confidence threshold
            valid_mask = scores >= self.confidence_threshold
            image_preds = image_preds[valid_mask]
            scores = scores[valid_mask]
            class_ids = class_ids[valid_mask]

            if image_preds.shape[0] == 0:
                outputs.append([])
                continue

            # Convert box format and apply NMS
            boxes = self._xywh_to_xyxy(image_preds[:, :4])
            keep_indices = torchvision.ops.nms(boxes, scores, self.iou_threshold)
            keep_indices = keep_indices[: self.max_det]

            detections = [
                {
                    "box": boxes[idx].tolist(),
                    "confidence": scores[idx].item(),
                    "class_id": class_ids[idx].item(),
                    "class_name": self.class_names[class_ids[idx]] if class_ids[idx] < len(self.class_names) else "unknown",
                }
                for idx in keep_indices
            ]

            outputs.append(detections)

        return outputs

    def rescale_detections(
        self,
        detections: List[List[Dict[str, object]]],
        input_size: Tuple[int, int],
        original_sizes: List[Tuple[int, int]]
    ) -> List[List[Dict[str, object]]]:
        """
        Rescale detections to original image sizes.

        Args:
            detections (List[List[Dict[str, object]]]): Detections from `apply_nms`.
            input_size (Tuple[int, int]): Model input size (width, height).
            original_sizes (List[Tuple[int, int]]): Original sizes of images (height, width).

        Returns:
            List[List[Dict[str, object]]]: Rescaled detections per image.
        """
        input_w, input_h = input_size
        rescaled_detections = []

        for i, image_detections in enumerate(detections):
            orig_h, orig_w = original_sizes[i]
            scale = min(input_w / orig_w, input_h / orig_h)
            pad_x = (input_w - orig_w * scale) / 2  # Padding along x-axis
            pad_y = (input_h - orig_h * scale) / 2  # Padding along y-axis

            rescaled_image_detections = []
            for det in image_detections:
                x1, y1, x2, y2 = det["box"]
                det["box"] = [
                    max(0, (x1 - pad_x) / scale),
                    max(0, (y1 - pad_y) / scale),
                    min(orig_w, (x2 - pad_x) / scale),
                    min(orig_h, (y2 - pad_y) / scale),
                ]
                rescaled_image_detections.append(det)

            rescaled_detections.append(rescaled_image_detections)

        return rescaled_detections
