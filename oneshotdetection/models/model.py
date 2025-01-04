import torch
import math
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple
import torch.nn as nn

from oneshotdetection.utils.model_utils import ModelUtils
from oneshotdetection.utils.image_processing import ImageProcessing
from oneshotdetection.utils.config_parser import ConfigParser
# TODO: split other file or process
from oneshotdetection.models.layers import Conv, DWConv, Detect


class BaseModel(nn.Module):
    """
    YOLOv5 Base Model with modular design for extensibility and maintainability.
    """

    def forward(self, x: torch.Tensor, profile: bool = False, visualize: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass of the model. Supports profiling and visualization.

        Args:
            x (torch.Tensor): Input tensor.
            profile (bool): Whether to profile each layer's performance.
            visualize (Optional[str]): Directory to save feature visualizations.

        Returns:
            torch.Tensor: Model output tensor.
        """
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x: torch.Tensor, profile: bool = False, visualize: Optional[str] = None) -> torch.Tensor:
        """
        Single-scale forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.
            profile (bool): Whether to profile each layer's performance.
            visualize (Optional[str]): Directory to save feature visualizations.

        Returns:
            torch.Tensor: Final output tensor.
        """
        outputs, timings = [], []
        for layer in self.model:
            if layer.f != -1:  # not from the previous layer
                x = outputs[layer.f] if isinstance(layer.f, int) else [x if i == -1 else outputs[i] for i in layer.f]

            if profile:
                timings.append(self._profile_layer(layer, x))

            x = layer(x)

            outputs.append(x if layer.i in self.save else None)

            if visualize:
                self._visualize_features(x, layer.type, layer.i, visualize)

        return x

    def _profile_layer(self, layer: nn.Module, x: torch.Tensor) -> float:
        """
        Profile a single layer's performance.

        Args:
            layer (nn.Module): The layer to profile.
            x (torch.Tensor): Input tensor.

        Returns:
            float: Time taken in milliseconds for a single pass.
        """
        import time
        start_time = time.perf_counter()
        _ = layer(x)
        return (time.perf_counter() - start_time) * 1000

    def fuse(self) -> "BaseModel":
        """
        Fuse Conv2d + BatchNorm2d layers for inference optimization.

        Returns:
            BaseModel: Self with fused layers.
        """
        for layer in self.model.modules():
            if isinstance(layer, (Conv, DWConv)) and hasattr(layer, "bn"):
                layer.conv = self._fuse_conv_and_bn(layer.conv, layer.bn)
                delattr(layer, "bn")
                layer.forward = layer.forward_fuse
        self.info()
        return self

    def _fuse_conv_and_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        """
        Fuse Conv2d and BatchNorm2d layers into a single Conv2d.

        Args:
            conv (nn.Conv2d): Convolutional layer.
            bn (nn.BatchNorm2d): Batch normalization layer.

        Returns:
            nn.Conv2d: Fused Conv2d layer.
        """
        with torch.no_grad():
            conv.weight = nn.Parameter(conv.weight * (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(-1, 1, 1, 1))
            conv.bias = nn.Parameter(bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps))
        return conv

    def initialize_weights(self) -> None:
        """
        Initialize weights for all layers in the model.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
                module.inplace = True

    def info(self, verbose: bool = False, img_size: int = 640) -> None:
        """
        Print model information.

        Args:
            verbose (bool): Whether to print detailed information.
            img_size (int): Input image size.
        """
        ModelUtils.model_info(self, verbose, img_size)

    def _apply(self, fn):
        """
        Apply transformations to model parameters and buffers.

        Args:
            fn: Transformation function.

        Returns:
            BaseModel: Transformed model.
        """
        self = super()._apply(fn)
        detect_layer = self._get_detection_layer()
        if detect_layer:
            detect_layer.stride = fn(detect_layer.stride)
            detect_layer.grid = [fn(grid) for grid in detect_layer.grid]
            if isinstance(detect_layer.anchor_grid, list):
                detect_layer.anchor_grid = [fn(ag) for ag in detect_layer.anchor_grid]
        return self

    def _get_detection_layer(self) -> Optional[nn.Module]:
        """
        Get the detection layer of the model.

        Returns:
            Optional[nn.Module]: Detection layer if present, else None.
        """
        return next((layer for layer in self.model if isinstance(layer, Detect)), None)


class DetectionModel(BaseModel):
    # Constants for bias initialization
    BASE_IMAGE_SIZE = 640  # Reference image size for scaling
    OBJECT_BIAS_FACTOR = 8  # Factor for objectness bias adjustment
    CLASS_BIAS_SCALE = 0.6  # Default scaling for class bias
    EPSILON = 1e-6  # Small value to prevent division errors
    OBJECTNESS_INDEX = 4  # Corresponds to the objectness score (index 4)
    CLASS_INDEX_START = 5  # Corresponds to class scores (indices 5 to end)

    # YOLOv5 detection model
    def __init__(self, cfg: str = 'yolov5s.yaml', ch: int = 3, nc: int = 80, anchors: Optional[Union[int, List]] = None) -> None:
        super().__init__()
        self.yaml = self._load_yaml(cfg)
        self._build_model(ch, nc, anchors)

        # Build strides, anchors
        self._build_strides_and_anchors(ch)

        # Init weights, biases
        self.initialize_weights()
        self.info()

    def _load_yaml(self, cfg: Union[str, Dict]) -> Dict:
        if isinstance(cfg, dict):
            return cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                return yaml.safe_load(f)

    def _build_model(
        self,
        input_channels: int,
        num_classes: int,
        anchors: Optional[Union[int, List[int]]]
    ) -> None:
        """
        Build the YOLO model by parsing the configuration.

        Args:
            input_channels (int): Number of input channels for the first layer.
            num_classes (int): Number of classes for detection.
            anchors (int or list): Anchor configuration to override.

        """
        # Resolve input channels
        self.yaml['ch'] = self._resolve_input_channels(input_channels)

        # Resolve number of classes
        self.yaml['nc'] = self._resolve_num_classes(num_classes)

        # Resolve anchors
        self.yaml['anchors'] = self._resolve_anchors(anchors)

        # Parse model configuration and construct the model
        self.model, self.save = ConfigParser.parse_model(deepcopy(self.yaml), [self.yaml['ch']])

        # Configure default names and inplace behavior
        self.names = [str(i) for i in range(self.yaml['nc'])]  # Default class names
        self.inplace = self.yaml.get('inplace', True)

    def _resolve_input_channels(self, input_channels: int) -> int:
        """
        Resolve the number of input channels for the first layer.

        This method ensures that the input channels are derived from the YAML configuration 
        or fall back to the provided value if not defined.

        Args:
            input_channels (int): Default input channels (e.g., 3 for RGB images).

        Returns:
            int: Resolved input channels, based on the configuration or the provided default.
        """
        # Default input channels in most YOLO models are 3 (for RGB images).
        DEFAULT_INPUT_CHANNELS = 3

        # Retrieve 'ch' from YAML or fallback to the default or provided input_channels.
        resolved_channels = self.yaml.get('ch', input_channels or DEFAULT_INPUT_CHANNELS)

        # Ensure the resolved value is an integer
        if not isinstance(resolved_channels, int):
            raise ValueError(f"Resolved input channels must be an integer, got {type(resolved_channels).__name__}")

        return resolved_channels

    def _resolve_num_classes(self, num_classes: int, default: int = 80) -> int:
        """
        Resolve the number of detection classes.

        This method determines the number of classes based on provided input or YAML configuration.
        If no valid input is found, it falls back to the default.

        Args:
            num_classes (int): Provided number of detection classes.
            default (int): Default number of classes to fall back to if not provided. Defaults to 80.

        Returns:
            int: Resolved number of detection classes.
        """
        # Default number of classes in COCO dataset is typically 80
        DEFAULT_NUM_CLASSES = default

        # Check if the provided number of classes is valid
        if num_classes and isinstance(num_classes, int):
            resolved_classes = num_classes
        else:
            # Fallback to YAML configuration or default
            resolved_classes = int(self.yaml.get('nc', DEFAULT_NUM_CLASSES))

        # Ensure the resolved value is a positive integer
        if resolved_classes <= 0:
            raise ValueError(f"Number of classes must be a positive integer, got {resolved_classes}")

        return resolved_classes

    def _resolve_anchors(self, anchors):
        """
        Resolve the anchor configuration.

        Args:
            anchors (int or list): Provided anchors configuration.

        Returns:
            int or list: Resolved anchors configuration.
        """
        if anchors:
            try:
                return round(anchors) if isinstance(anchors, int) else anchors
            except Exception as e:
                print(f"Error resolving anchors: {e}. Using default anchors.")
        return self.yaml.get('anchors', None)

    def _build_strides_and_anchors(self, input_channels):
        """
        Build strides and anchors for the detection layer.
        """
        # Get the detection layer
        detection_layer = self._get_detection_layer()

        if isinstance(detection_layer, Detect):
            default_input_size = 256  # Default input size for stride calculation
            dummy_input = torch.zeros(1, input_channels, default_input_size, default_input_size)
            
            # Calculate strides
            detection_layer.stride = torch.tensor(
                [default_input_size / x.shape[-2] for x in self.forward(dummy_input)]
            )
            
            # Ensure anchor order matches stride order
            self._check_anchor_order(detection_layer)
            
            # Normalize anchors by stride
            detection_layer.anchors /= detection_layer.stride.view(-1, 1, 1)
            
            # Store stride
            self.stride = detection_layer.stride
            
            # Initialize biases
            self._initialize_biases(detection_layer)

    def _get_detection_layer(self):
        """
        Get the last layer of the model, assumed to be a Detect() layer.

        Returns:
            Detect: The Detect() layer if found, None otherwise.
        """
        last_layer = self.model[-1]
        if isinstance(last_layer, Detect):
            return last_layer
        print("Warning: No Detect() layer found in the model.")
        return None

    def _check_and_adjust_anchor_order(self, detection_layer):
        """
        Check and adjust anchor order to match stride order in the Detect() layer.

        Args:
            detection_layer (Detect): The Detect() layer.
        """
        mean_anchor_area = detection_layer.anchors.prod(-1).mean(-1).view(-1)
        delta_area = mean_anchor_area[-1] - mean_anchor_area[0]
        delta_stride = detection_layer.stride[-1] - detection_layer.stride[0]

        if delta_area and (delta_area.sign() != delta_stride.sign()):
            print("Reversing anchor order to match stride order.")
            detection_layer.anchors[:] = detection_layer.anchors.flip(0)

    def _check_anchor_order(self, detection_layer):
        """
        Check and correct anchor order for the Detect() module.

        Args:
            detection_layer (Detect): The detection layer instance.
        """
        anchor_area = detection_layer.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area
        delta_anchor_area = anchor_area[-1] - anchor_area[0]  # delta of anchor areas
        delta_stride = detection_layer.stride[-1] - detection_layer.stride[0]  # delta of strides

        if delta_anchor_area and (delta_anchor_area.sign() != delta_stride.sign()):
            print("Reversing anchor order for consistency with stride order")
            detection_layer.anchors[:] = detection_layer.anchors.flip(0)

    def _initialize_biases(self, detection_layer, class_frequency=None):
        """
        Initialize biases for the detection layer.

        Args:
            detection_layer (Detect): The detection layer instance.
            class_frequency (Tensor, optional): Class frequency tensor. Defaults to None.
        """
        for module, stride in zip(detection_layer.m, detection_layer.stride):
            # Retrieve current biases
            bias = module.bias.view(detection_layer.na, -1).detach()

            # Adjust objectness bias
            bias[:, self.OBJECTNESS_INDEX] += math.log(self.OBJECT_BIAS_FACTOR / (self.BASE_IMAGE_SIZE / stride) ** 2)

            # Adjust class bias
            if class_frequency is None:
                bias[:, self.CLASS_INDEX_START:] += math.log(self.CLASS_BIAS_SCALE / (detection_layer.nc - 1 + self.EPSILON))
            else:
                bias[:, self.CLASS_INDEX_START:] += torch.log(class_frequency / class_frequency.sum())

            # Update the module bias
            module.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

    def _get_detection_layer(self):
        """
        Search for the Detect() layer in the model.

        Returns:
            Detect: The Detect() layer if found, None otherwise.
        """
        for layer in self.model.modules():
            if isinstance(layer, Detect):
                return layer
        print("Warning: No Detect() layer found in the model.")
        return None

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[None]]:
        """
        Perform augmented forward pass for YOLOv5.

        This method applies scale and flip augmentations to the input tensor, processes each augmentation through
        the model, and merges the results.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where:
                            - N: Batch size
                            - C: Number of channels
                            - H: Height
                            - W: Width

        Returns:
            Tuple[torch.Tensor, Optional[None]]:
                - Concatenated augmented outputs along dimension 1 (e.g., predictions from different scales/flips).
                - None (reserved for compatibility with training pipeline).
        """
        img_size = x.shape[-2:]  # Input image dimensions: (height, width)
        scale_factors = [1.0, 0.83, 0.67]  # Scaling factors: original, medium, and small
        flip_modes = [None, 3, None]  # Flip modes: None, horizontal (3), None
        augmented_outputs: List[torch.Tensor] = []  # List to store outputs from each augmentation

        for scale_factor, flip_mode in zip(scale_factors, flip_modes):
            # Apply flip if specified, followed by scaling
            augmented_input = ImageProcessing.scale_img(
                x.flip(flip_mode) if flip_mode else x, scale_factor, gs=int(self.stride.max())
            )
            # Perform forward pass on augmented input
            augmented_output = self._forward_once(augmented_input)[0]
            # Descale the predictions to match original scale and orientation
            descaled_output = self._descale_pred(augmented_output, flip_mode, scale_factor, img_size)
            augmented_outputs.append(descaled_output)

        # Clip augmented outputs to handle edges
        final_output = self._clip_augmented(augmented_outputs)

        # Concatenate results from all augmentations along dimension 1
        return torch.cat(final_output, dim=1), None  # Augmented inference results

    def _descale_pred(
        self,
        predictions: torch.Tensor,
        flips: Optional[int],
        scale: float,
        img_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        De-scale predictions following augmented inference (inverse operation).

        Args:
            predictions (torch.Tensor): Predictions tensor with shape (..., num_features).
            flips (Optional[int]): Flip mode, 2 for vertical flip, 3 for horizontal flip, None for no flip.
            scale (float): Scale factor applied during augmentation.
            img_size (Tuple[int, int]): Original image size (height, width).

        Returns:
            torch.Tensor: De-scaled predictions.
        """
        # Define feature indices for better readability
        BBOX_COORDS = slice(0, 4)  # x, y, width, height
        CENTER_X = slice(0, 1)     # x center coordinate
        CENTER_Y = slice(1, 2)     # y center coordinate
        WH = slice(2, 4)           # width and height
        REMAINING_FEATURES = slice(4, None)  # remaining features, e.g., confidence and class scores

        if self.inplace:
            # Apply inverse scaling directly
            predictions[..., BBOX_COORDS] /= scale

            # Apply de-flipping
            if flips == 2:  # Vertical flip
                predictions[..., 1] = img_size[0] - predictions[..., 1]  # De-flip y
            elif flips == 3:  # Horizontal flip
                predictions[..., 0] = img_size[1] - predictions[..., 0]  # De-flip x
        else:
            # Separate coordinates for explicit transformation
            x = predictions[..., CENTER_X] / scale
            y = predictions[..., CENTER_Y] / scale
            wh = predictions[..., WH] / scale

            # Apply de-flipping
            if flips == 2:  # Vertical flip
                y = img_size[0] - y
            elif flips == 3:  # Horizontal flip
                x = img_size[1] - x

            # Combine transformed components back
            predictions = torch.cat((x, y, wh, predictions[..., REMAINING_FEATURES]), -1)

        return predictions

    def _clip_augmented(self, predictions: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Clip augmented YOLOv5 inference tails.

        Args:
            predictions (List[torch.Tensor]): List of predictions from augmented inference.

        Returns:
            List[torch.Tensor]: Clipped predictions.
        """
        # Constants for clarity
        DETECTION_LAYERS = self.model[-1].nl  # Number of detection layers (e.g., P3, P4, P5)
        GRID_SCALING_BASE = 4  # Base for grid scaling
        EXCLUDE_LAYER_COUNT = 1  # Number of layers to exclude in clipping
        
        # Calculate total grid points
        total_grid_points = sum(GRID_SCALING_BASE ** layer_index for layer_index in range(DETECTION_LAYERS))
        
        # Clip large tails for the first layer
        first_layer_scaling = sum(GRID_SCALING_BASE ** layer_index for layer_index in range(EXCLUDE_LAYER_COUNT))
        first_layer_indices = (predictions[0].shape[1] // total_grid_points) * first_layer_scaling
        predictions[0] = predictions[0][:, :-first_layer_indices]  # Clip large tails
        
        # Clip small tails for the last layer
        last_layer_scaling = sum(
            GRID_SCALING_BASE ** (DETECTION_LAYERS - 1 - layer_index) for layer_index in range(EXCLUDE_LAYER_COUNT)
        )
        last_layer_indices = (predictions[-1].shape[1] // total_grid_points) * last_layer_scaling
        predictions[-1] = predictions[-1][:, last_layer_indices:]  # Clip small tails
        
        return predictions
