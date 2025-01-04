import math
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple


def calculate_padding(kernel_size: int, padding: Optional[int] = None) -> int:
    """
    Calculate padding for a given kernel size if padding is not explicitly provided.

    Args:
        kernel_size (int): Size of the convolution kernel.
        padding (Optional[int]): Provided padding, if any.

    Returns:
        int: Calculated padding size.
    """
    if padding is not None:
        return padding
    return kernel_size // 2  # Default padding to keep output dimensions consistent


class Conv(nn.Module):
    """
    Standard convolutional layer with optional activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        padding (Optional[int]): Padding for the convolution. If None, calculated automatically.
        groups (int): Number of groups for the convolution.
        activation (Union[bool, nn.Module]): Activation function or flag to use SiLU (default: True).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: Union[bool, nn.Module] = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=calculate_padding(kernel_size, padding),
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation is True:
            self.act = nn.SiLU()  # Default activation function
        elif isinstance(activation, nn.Module):
            self.act = activation
        else:
            self.act = nn.Identity()  # No activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for fused layers (without BatchNorm).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    """
    Depth-wise convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        activation (Union[bool, nn.Module]): Activation function or flag to use SiLU (default: True).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        activation: Union[bool, nn.Module] = True
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=math.gcd(in_channels, out_channels),  # Use greatest common divisor for groups
            activation=activation
        )


class Bottleneck(nn.Module):
    """
    Standard bottleneck with a configurable shortcut.
    """
    def __init__(self, in_channels: int, out_channels: int, use_shortcut: bool = True, 
                 groups: int = 1, expansion: float = 0.5) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_shortcut (bool): Whether to use a residual shortcut. Defaults to True.
            groups (int): Number of groups for group convolution. Defaults to 1.
            expansion (float): Expansion factor for hidden channels. Defaults to 0.5.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # Hidden channels
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1, groups=groups)
        self.use_shortcut = use_shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Bottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.use_shortcut:
            return x + self.cv2(self.cv1(x))
        return self.cv2(self.cv1(x))


class C3(nn.Module):
    """
    CSP Bottleneck with 3 convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, num_repeats: int = 1, 
                 use_shortcut: bool = True, groups: int = 1, expansion: float = 0.5) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_repeats (int): Number of Bottleneck layers in the module.
            use_shortcut (bool): Whether bottlenecks use residual shortcuts.
            groups (int): Number of groups for group convolution.
            expansion (float): Expansion factor for hidden channels.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # Hidden channels
        
        # Define the layers
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv3 = Conv(hidden_channels * 2, out_channels, kernel_size=1)  # Combine
        
        # Construct the sequential block of bottlenecks
        self.m = nn.Sequential(
            *(Bottleneck(hidden_channels, hidden_channels, use_shortcut, groups, expansion=1.0)
              for _ in range(num_repeats))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the C3 layer.
        """
        # Branch 1: cv1 + m
        y1 = self.m(self.cv1(x))
        
        # Branch 2: cv2
        y2 = self.cv2(x)
        
        # Concatenate and pass through cv3
        return self.cv3(torch.cat((y1, y2), dim=1))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along a specified dimension.
    """
    def __init__(self, dimension: int = 1):
        """
        Initialize the Concat layer.

        Args:
            dimension (int): The dimension along which to concatenate tensors. Default is 1.
        """
        super().__init__()
        self.dimension = dimension

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass: concatenate a list of tensors along the specified dimension.

        Args:
            inputs (list[torch.Tensor]): List of tensors to concatenate.

        Returns:
            torch.Tensor: A single concatenated tensor.
        """
        if not all(isinstance(tensor, torch.Tensor) for tensor in inputs):
            raise ValueError("All inputs must be of type torch.Tensor.")
        return torch.cat(inputs, dim=self.dimension)


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer.
    Efficient feature aggregation using multiple max-pooling operations.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 5):
        """
        Initialize the SPPF layer.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            kernel_size (int): Kernel size for max-pooling. Defaults to 5.
        """
        super().__init__()
        hidden_channels = input_channels // 2  # Reduce channels for intermediate processing

        self.cv1 = Conv(input_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = Conv(hidden_channels * 4, output_channels, kernel_size=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SPPF layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after spatial pyramid pooling.
        """
        x1 = self.cv1(x)  # Reduce channels
        y1 = self.maxpool(x1)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.cv2(torch.cat([x1, y1, y2, y3], dim=1))  # Concatenate and project


class Detect(nn.Module):
    """
    Detection layer for YOLO models.
    """
    stride: List[float] = None  # strides computed during build
    onnx_dynamic: bool = False  # ONNX export parameter
    export: bool = False  # export mode

    def __init__(
        self, 
        nc: int = 80, 
        anchors: List[List[float]] = (), 
        ch: List[int] = (), 
        inplace: bool = True
    ):
        """
        Initialize the detection layer.

        Args:
            nc (int): Number of classes.
            anchors (List[List[float]]): Anchor configurations.
            ch (List[int]): List of input channels for each detection layer.
            inplace (bool): Use inplace operations.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.inplace = inplace

        # Initialize grids
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl

        # Register anchors as buffer
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))

        # Output convolution layers
        self.m = nn.ModuleList([nn.Conv2d(c, self.no * self.na, kernel_size=1) for c in ch])

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for the detection layer.

        Args:
            x (List[torch.Tensor]): Input feature maps.

        Returns:
            Tuple[torch.Tensor, ...]: Processed feature maps for training or inference.
        """
        outputs = []  # List to store outputs during inference
        for i in range(self.nl):
            # Apply convolution
            x[i] = self.m[i](x[i])  # bs, (na * no), ny, nx
            bs, _, ny, nx = x[i].shape

            # Reshape to (batch, anchors, grid_y, grid_x, outputs)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference mode
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    # Adjust predictions for xy, wh
                    y[..., :2] = (y[..., :2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), dim=4)
                    xy = (xy * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, conf), dim=4)

                # Append results for inference
                outputs.append(y.view(bs, -1, self.no))

        # Return outputs for inference or training
        return (torch.cat(outputs, dim=1),) if not self.training else x

    def _make_grid(self, nx: int, ny: int, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create the grid and anchor grid for a detection layer.

        Args:
            nx (int): Grid width.
            ny (int): Grid height.
            i (int): Index of the detection layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Grid and anchor grid tensors.
        """
        device = self.anchors.device
        dtype = self.anchors.dtype

        # Generate grid
        y, x = torch.arange(ny, device=device, dtype=dtype), torch.arange(nx, device=device, dtype=dtype)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), dim=2).expand(1, self.na, ny, nx, 2)

        # Generate anchor grid
        anchor_grid = (self.anchors[i] * self.stride[i]).view(1, self.na, 1, 1, 2).expand_as(grid)

        return grid, anchor_grid
