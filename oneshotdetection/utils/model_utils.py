import torch
from pathlib import Path
from typing import Union
from copy import deepcopy
from torch import nn
from typing import Optional
from torch.serialization import add_safe_globals
import os


class ModelUtils:
    @staticmethod
    def model_info(model: nn.Module, verbose: bool = False, img_size: Union[int, list] = 640) -> None:
        """
        Display model information including number of layers, parameters, and FLOPs.

        Args:
            model (nn.Module): Model instance.
            verbose (bool): Whether to print detailed information for each layer.
            img_size (Union[int, list]): Input image size, either as a single integer or [height, width].
        """
        num_parameters = sum(param.numel() for param in model.parameters())  # Total parameters
        num_gradients = sum(param.numel() for param in model.parameters() if param.requires_grad)  # Trainable parameters

        # Verbose output for each parameter
        if verbose:
            print(
                f"{'Layer':>5} {'Name':>40} {'Gradient':>9} {'Parameters':>12} "
                f"{'Shape':>20} {'Mean':>10} {'Std':>10}"
            )
            for i, (name, param) in enumerate(model.named_parameters()):
                name = name.replace("module_list.", "")
                print(
                    f"{i:5d} {name:40s} {str(param.requires_grad):>9s} {param.numel():12d} "
                    f"{str(list(param.shape)):>20s} {param.mean():10.3g} {param.std():10.3g}"
                )

        # FLOPs calculation
        flops_str = ModelUtils._calculate_flops(model, img_size)
        
        # Model summary
        name = (
            Path(model.yaml_file).stem.replace("yolov5", "YOLOv5") if hasattr(model, "yaml_file") else "Model"
        )
        print(
            f"{name} summary: {len(list(model.modules()))} layers, {num_parameters} parameters, "
            f"{num_gradients} gradients{flops_str}"
        )

    @staticmethod
    def _calculate_flops(model: nn.Module, img_size: Union[int, list]) -> str:
        """
        Calculate FLOPs for the model.

        Args:
            model (nn.Module): Model instance.
            img_size (Union[int, list]): Input image size, either as a single integer or [height, width].

        Returns:
            str: FLOPs string to append to the model summary.
        """
        try:
            input_size = img_size if isinstance(img_size, list) else [img_size, img_size]
            p = next(model.parameters())
            stride = getattr(model, "stride", torch.tensor([32])).max().item()  # Default max stride to 32
            dummy_input = torch.zeros((1, p.shape[1], stride, stride), device=p.device)  # Input tensor (BCHW)

            # Compute FLOPs using torch's operations
            total_flops = Utils._profile_flops(model, dummy_input)
            gflops = total_flops / 1e9 * 2  # Convert to GFLOPs

            return f", {gflops * input_size[0] / stride * input_size[1] / stride:.1f} GFLOPs"
        except Exception:
            return ""

    @staticmethod
    def _profile_flops(model: nn.Module, dummy_input: torch.Tensor) -> float:
        """
        Compute FLOPs using a custom profiling method.

        Args:
            model (nn.Module): Model instance.
            dummy_input (torch.Tensor): Dummy input for FLOPs calculation.

        Returns:
            float: Total FLOPs for the model.
        """
        total_flops = 0.0
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                # FLOPs for Conv2d: Output elements * kernel elements * input channels
                output_elements = dummy_input.numel() // dummy_input.shape[1]
                kernel_elements = layer.weight.numel()
                total_flops += output_elements * kernel_elements
            elif isinstance(layer, nn.Linear):
                # FLOPs for Linear: Input elements * output elements
                total_flops += layer.in_features * layer.out_features
            # Add more layer types as needed

        return total_flops

    @staticmethod
    def output_model_info(model: nn.Module, file_path: str):
        with open(file_path, "w") as f:
            f.write(str(model))

    @staticmethod
    def remap_state_dict_keys(state_dict):
        """
        Adjust state_dict keys to match the current model structure.

        Args:
            state_dict (dict): Original state_dict.

        Returns:
            dict: Remapped state_dict compatible with the current model.
        """
        remapped_state_dict = {}

        for key, value in state_dict.items():
            # Example mapping: Adjust prefixes and internal names
            new_key = key
            if "model.conv1" in key:
                new_key = key.replace("model.conv1", "model.0.conv")
            elif "model.conv2" in key:
                new_key = key.replace("model.conv2", "model.1.conv")
            elif "model.c3" in key:  # Handle C3 layer differences
                new_key = key.replace("model.c3", "model.2.bottlenecks")
            # Add other rules as needed...

            remapped_state_dict[new_key] = value

        return remapped_state_dict

    @staticmethod
    def remap_and_save_state_dict(input_pt_file, output_pt_file):
        """
        Load, remap, and save the remapped state_dict to a new .pt file.

        Args:
            input_pt_file (str): Path to the original .pt file.
            output_pt_file (str): Path to save the remapped .pt file.
        """
        checkpoint = torch.load(input_pt_file, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise ValueError("The file does not contain a valid 'state_dict'.")

        state_dict = checkpoint["state_dict"]
        remapped_state_dict = ModelUtils.remap_state_dict_keys(state_dict)

        # Save the remapped state_dict
        torch.save({"state_dict": remapped_state_dict}, output_pt_file)
        print(f"Remapped state_dict saved to {output_pt_file}")

    @staticmethod
    def load_model_safely(pt_file: str, cfg_path: str, device: torch.device):
        """
        Load a YOLO model from a .pt file containing a pre-saved state_dict.

        Args:
            pt_file (str): Path to the .pt file.
            cfg_path (str): Path to the YAML configuration file.
            device (torch.device): Device to load the model onto.

        Returns:
            DetectionModel: Loaded model instance with weights.
        """
        from oneshotdetection.models.model import DetectionModel

        try:
            # Load the original state_dict
            checkpoint = torch.load(pt_file, map_location=device)
            if "state_dict" not in checkpoint:
                raise ValueError("The file does not contain a valid 'state_dict'.")

            state_dict = checkpoint["state_dict"]

            # Remap the state_dict keys
            remapped_state_dict = ModelUtils.remap_state_dict_keys(state_dict)

            # Initialize the model with cfg_path
            model = DetectionModel(cfg_path)

            # Load weights into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

            # Log mismatches
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

            print("Model weights loaded successfully.")

            # Add `input_dtype` to the model based on its parameters
            param_dtype = next(model.parameters()).dtype
            model.input_dtype = param_dtype

            print(f"Model loaded. Precision: {'float16' if param_dtype == torch.float16 else 'float32'}")

            return model.to(device).eval()
        except Exception as e:
            print(f"Error while loading the model: {e}")
            raise
