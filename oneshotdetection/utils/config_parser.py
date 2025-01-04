import math
import torch
from torch import nn
import contextlib
from oneshotdetection.models.layers import Conv, C3, Bottleneck, Concat, SPPF, Detect


DETECTION_BBOX_PARAMS = 5  # Number of parameters for bounding boxes (x, y, w, h, confidence)

class ConfigParser:
    """
    Utility class for parsing YOLO model configurations and building model layers.
    """

    @staticmethod
    def parse_model(config, input_channels):
        """
        Parse model configuration and construct model layers sequentially.

        Args:
            config (dict): Model configuration loaded from YAML or dictionary.
            input_channels (list): List of input channels for each layer.

        Returns:
            nn.Sequential: Constructed model layers as a sequential container.
            list: Indices of layers to save outputs.
        """
        print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")

        # Extract model parameters
        anchors = config['anchors']
        num_classes = config['nc']
        depth_multiplier = config['depth_multiple']
        width_multiplier = config['width_multiple']

        # Compute anchor and output sizes
        num_anchors = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
        output_channels = num_anchors * (num_classes + DETECTION_BBOX_PARAMS)

        # Initialize model layers and metadata
        layers = []
        save_indices = []
        current_output_channels = input_channels[-1]

        # Layer parsing strategy dictionary
        layer_strategies = {
            'common': ConfigParser._handle_common_module,
            'detect': ConfigParser._handle_detect_module,
            'concat': ConfigParser._handle_concat_module,
        }

        # Iterate over backbone and head configurations
        for layer_index, (from_layer, num_repeats, module_type, arguments) in enumerate(config['backbone'] + config['head']):
            module_class = eval(module_type) if isinstance(module_type, str) else module_type

            # Resolve arguments (if strings)
            for arg_index, arg in enumerate(arguments):
                with contextlib.suppress(NameError):
                    arguments[arg_index] = eval(arg) if isinstance(arg, str) else arg

            # Adjust number of repeats
            num_repeats = max(round(num_repeats * depth_multiplier), 1) if num_repeats > 1 else num_repeats

            # Determine input channels
            input_channel_count = (
                input_channels[from_layer]
                if isinstance(from_layer, int)
                else sum(input_channels[idx] for idx in from_layer if idx < len(input_channels))
            )

            # Handle specific module types
            if module_class in (Conv, Bottleneck, SPPF, C3):
                current_output_channels, arguments = layer_strategies['common'](
                    arguments, input_channel_count, current_output_channels, width_multiplier, output_channels, module_class, num_repeats
                )
                if module_class in [C3]:  # Insert repeats for C3
                    arguments.insert(2, num_repeats)
                num_repeats = 1  # Reset repeat count
            elif module_class is Detect:
                arguments = layer_strategies['detect'](arguments, num_classes, input_channels, from_layer)
            elif module_class is Concat:
                current_output_channels = layer_strategies['concat'](input_channels, from_layer)

            # Build the module
            layer_instance = (
                nn.Sequential(*(module_class(*arguments) for _ in range(num_repeats)))
                if num_repeats > 1
                else module_class(*arguments)
            )

            # Attach metadata
            module_type_str = str(module_class)[8:-2].replace('__main__.', '')
            num_parameters = sum(param.numel() for param in layer_instance.parameters())
            layer_instance.i = layer_index
            layer_instance.f = from_layer
            layer_instance.type = module_type_str
            layer_instance.np = num_parameters

            # Print layer information
            print(
                f"{layer_index:>3}{str(from_layer):>18}{num_repeats:>3}{num_parameters:10.0f}  "
                f"{module_type_str:<40}{str(arguments):<30}"
            )

            # Update save indices and add the layer
            save_indices.extend(
                index % layer_index for index in ([from_layer] if isinstance(from_layer, int) else from_layer) if index != -1
            )
            layers.append(layer_instance)

            # Update input channel list
            if layer_index == 0:
                input_channels = []
            input_channels.append(current_output_channels)

        return nn.Sequential(*layers), sorted(save_indices)

    @staticmethod
    def _handle_common_module(args, input_channels, current_output_channels, width_multiplier, output_channels, module_class, num_repeats):
        """
        Handle common modules like Conv, Bottleneck, SPPF, and C3.
        """
        new_output_channels = args[0]
        if new_output_channels != output_channels:  # Adjust output channels for non-output layers
            new_output_channels = ConfigParser.make_divisible(new_output_channels * width_multiplier, 8)
        args = [input_channels, new_output_channels, *args[1:]]
        return new_output_channels, args

    @staticmethod
    def _handle_detect_module(args, num_classes, input_channels, from_layer):
        """
        Handle the Detect module.
        """
        args = [num_classes if a == 'nc' else a for a in args]
        args.append([input_channels[idx] for idx in from_layer])
        if isinstance(args[1], int):  # Adjust anchor sizes
            args[1] = [list(range(args[1] * 2))] * len(from_layer)
        return args

    @staticmethod
    def _handle_concat_module(input_channels, from_layer):
        """
        Handle the Concat module.
        """
        return sum(input_channels[idx] for idx in from_layer)

    @staticmethod
    def make_divisible(value: float, divisor: int) -> int:
        """
        Adjusts a given value to the nearest value that is divisible by the divisor.

        Args:
            value (float): The input value to adjust.
            divisor (int): The divisor to align with.

        Returns:
            int: The nearest value that is divisible by the divisor.
        """
        if isinstance(divisor, torch.Tensor):
            divisor = int(divisor.max().item())
        adjusted_value = math.ceil(value / divisor) * divisor
        return max(adjusted_value, divisor)
