import torch
import yaml
import argparse
from oneshotdetection.models.model import DetectionModel
from oneshotdetection.utils.model_utils import ModelUtils
from oneshotdetection.utils.preprocessor import Preprocessor
from oneshotdetection.utils.postprocessor import PostProcessor
from oneshotdetection.utils.visualizer import Visualizer
from oneshotdetection.utils.model_utils import ModelUtils

import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_hyp(hyp_yml):
    hyp = {}
    with open(hyp_yml, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
        if 'anchors' not in hyp:  # anchors commented in hyp.yaml
            hyp['anchors'] = 3
    return hyp


def process_model_file_load(
        pt_file = "bin/yolov5s_weights.pt",
        cfg_path = "configs/yolov5s.yaml",
        input_shape = (640, 640)
        ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ModelUtils.remap_and_save_state_dict(pt_file, "remapped_yolov5s.pt")
    model = ModelUtils.load_model_safely("remapped_yolov5s.pt", cfg_path, device)
    model.to(device).eval()  # Move model to the appropriate device and set to evaluation mode

    # Prepare dummy input
    dummy_input = torch.zeros((1, 3, input_shape[0], input_shape[1])).to(device)
    dummy_input = dummy_input.to(dtype=model.input_dtype, device=device)
    # warmup
    model(dummy_input)

    # Forward pass
    # output = model(dummy_input)

    # Process output
    # inspect_shapes(output, prefix="Feature Map")
    return model


def inspect_shapes(output, prefix="Output"):
    """
    Recursively inspect and print shapes of tensors in a model output.
    
    Args:
        output: The model output, which can be a Tensor, list, tuple, or nested structure.
        prefix (str): Prefix for logging the output shape or type.
    """
    if isinstance(output, torch.Tensor):
        print(f"{prefix}: shape {output.shape}")
    elif isinstance(output, (list, tuple)):
        for i, item in enumerate(output):
            inspect_shapes(item, prefix=f"{prefix}[{i}]")
    else:
        print(f"{prefix}: type {type(output)} - Cannot retrieve shape")


def process_model_build(
        cfg = "config/size_s.yaml",
        nc = 80,
        hyp_yml = "config/hyp_low.yaml",
        input_shape = (640, 640)
        ):
    hyp = load_hyp(hyp_yml)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DetectionModel(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    # warmup
    dummy_input = torch.zeros((1, 3, input_shape[0], input_shape[1])).to(device)
    model(dummy_input)
    # output = model(dummy_input)

    # for i, o in enumerate(output):
    #     print(f"Feature Map {i}: shape {o.shape}")

    return model


def main():
    """
    Entry point for running the OneShotDetection module.
    """
    parser = argparse.ArgumentParser(description="Run One Shot Detection pipeline.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model weights (e.g., .pt file).")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config file (e.g., .yaml).")
    parser.add_argument("--input-shape", type=int, nargs=2, default=(640, 640), help="Input shape (height, width).")
    parser.add_argument("--output-dir", type=str, default="runs", help="Directory to save visualized results.")
    parser.add_argument("--output-file", type=str, default="result.jpg", help="Output file name for visualized results.")
    args = parser.parse_args()

    # Load model
    model = process_model_file_load(
        pt_file=args.model,
        cfg_path=args.config,
        input_shape=args.input_shape
    )

    # Preprocess image
    preprocessor = Preprocessor(input_shape=args.input_shape)
    image = preprocessor.load_image(args.image)
    tensor, original_size, (scale, padding) = preprocessor.preprocess(image)

    # Move tensor to model device
    tensor = tensor.to(next(model.parameters()).device)

    # Inference
    output = model(tensor)

    # Postprocess
    postprocessor = PostProcessor()
    predictions = output[0]
    detections = postprocessor.apply_nms(predictions)
    rescaled_detections = postprocessor.rescale_detections(
        detections, input_size=args.input_shape, original_sizes=[original_size]
    )

    # Print detections
    print("=" * 100)
    for idx, detection in enumerate(rescaled_detections[0]):
        print(f"{idx}: {detection}")

    # Visualize and save results
    visualizer = Visualizer(output_dir=args.output_dir)
    visualizer.visualize_and_save(image, rescaled_detections[0], args.output_file)
    print("=" * 100)


if __name__ == "__main__":
    main()
