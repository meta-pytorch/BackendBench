#!/usr/bin/env python3

"""
Toy model that uses core PyTorch operators during training.

This model is designed to trigger all of the following backward operators
when performing backpropagation:

- ConvolutionBackward0          (convolution_backward)
- NativeGroupNormBackward0      (native_group_norm_backward)
- MaxPool2DWithIndicesBackward0 (max_pool2d_with_indices_backward)
- AvgPool2DBackward0            (avg_pool2d_backward)
- AdaptiveAvgPool2DBackward0    (_adaptive_avg_pool2d_backward)

The model implements a CNN architecture with the following structure:
1. Conv2d -> GroupNorm -> ReLU           (triggers convolution_backward, native_group_norm_backward)
2. MaxPool2d with indices                (triggers max_pool2d_with_indices_backward)
3. Conv2d -> GroupNorm -> ReLU           (triggers convolution_backward, native_group_norm_backward again)
4. AvgPool2d                             (triggers avg_pool2d_backward)
5. AdaptiveAvgPool2d                     (triggers _adaptive_avg_pool2d_backward)
6. Final Conv2d                          (triggers convolution_backward again)

Usage:
    python toy_core_ops.py

This will create a model with default configuration and run a simple forward/backward pass
to demonstrate that all required backward operators are used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyCoreOpsModel(nn.Module):
    """
    Toy CNN model designed to test core PyTorch operators during training.

    The model uses a strategic combination of operations to ensure all target
    backward operators are invoked during backpropagation:

    - Convolution layers for convolution_backward
    - Group normalization for native_group_norm_backward
    - Max pooling with indices for max_pool2d_with_indices_backward
    - Average pooling for avg_pool2d_backward
    - Adaptive average pooling for _adaptive_avg_pool2d_backward
    """

    def __init__(self,
                 in_channels: int = 3,
                 hidden_channels: int = 32,
                 out_channels: int = 8,
                 num_groups: int = 8,
                 seed: int = 42):
        """
        Initialize the ToyCoreOpsModel.

        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            hidden_channels: Number of hidden channels in conv layers
            out_channels: Number of output channels
            num_groups: Number of groups for GroupNorm (must divide hidden_channels)
            seed: Random seed for deterministic weight initialization

        Raises:
            ValueError: If hidden_channels is not divisible by num_groups
        """
        super().__init__()

        # Validate group normalization constraints
        if hidden_channels % num_groups != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible "
                f"by num_groups ({num_groups})"
            )

        # Store configuration
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        # First convolution block (triggers convolution_backward)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1
        )

        # First group normalization (triggers native_group_norm_backward)
        self.group_norm1 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=hidden_channels
        )

        # Second convolution block (triggers convolution_backward again)
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1
        )

        # Second group normalization (triggers native_group_norm_backward again)
        self.group_norm2 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=hidden_channels
        )

        # Final convolution for output (triggers convolution_backward again)
        self.conv_out = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        # Initialize weights deterministically
        self._initialize_weights(seed)

    def _initialize_weights(self, seed: int):
        """
        Initialize model weights deterministically using the given seed.

        Args:
            seed: Random seed for reproducible initialization
        """
        # Set random seed for deterministic initialization
        torch.manual_seed(seed)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that sets up the computational graph to trigger all target backward operators.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, 4, 4)

        Note:
            The output is always 4x4 regardless of input size due to the adaptive pooling layer.
        """

        # First conv block: Conv2d -> GroupNorm -> ReLU
        # This will trigger: ConvolutionBackward0, NativeGroupNormBackward0
        x = self.conv1(x)
        x = self.group_norm1(x)
        x = F.relu(x)

        # Max pooling with indices (triggers MaxPool2DWithIndicesBackward0)
        # We need to use return_indices=True to get the specific backward operator
        x, indices = F.max_pool2d(x, kernel_size=2, return_indices=True)

        # Second conv block: Conv2d -> GroupNorm -> ReLU
        # This will trigger: ConvolutionBackward0, NativeGroupNormBackward0 (again)
        x = self.conv2(x)
        x = self.group_norm2(x)
        x = F.relu(x)

        # Average pooling (triggers AvgPool2DBackward0)
        x = F.avg_pool2d(x, kernel_size=2)

        # Adaptive average pooling (triggers AdaptiveAvgPool2DBackward0)
        # This ensures consistent output size regardless of input dimensions
        x = F.adaptive_avg_pool2d(x, output_size=(4, 4))

        # Final convolution (triggers ConvolutionBackward0 again)
        x = self.conv_out(x)

        return x


def main():
    """
    Demonstrate the ToyCoreOpsModel with a simple forward/backward pass.
    """
    print("ToyCoreOpsModel Demonstration")
    print("=" * 50)

    # Create model with default configuration
    model = ToyCoreOpsModel(
        in_channels=3,
        hidden_channels=32,
        out_channels=8,
        num_groups=8,
        seed=42  # Deterministic initialization
    )

    # Create sample input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 64, 64, requires_grad=True)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Input shape: {input_tensor.shape}")

    # Forward pass
    model.train()
    output = model(input_tensor)
    expected_shape = torch.Size([batch_size, 8, 4, 4])  # Expected output shape

    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {expected_shape}")
    print(f"Shape matches: {output.shape == expected_shape}")

    # Perform backward pass to actually trigger the operations
    print("\nPerforming backward pass...")
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass completed successfully")

    # Check gradients were computed
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = len(list(model.parameters()))
    print(f"✓ Gradients computed for {grad_count}/{total_params} parameters")

    print(f"\n✓ Model demonstration completed successfully!")
    print("This model is ready to be used with the Model Suite for testing core operators.")

    return model


if __name__ == "__main__":
    main()