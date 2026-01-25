# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
CNN model that triggers core PyTorch backward operators:
- convolution_backward
- native_group_norm_backward
- max_pool2d_with_indices_backward
- avg_pool2d_backward
- _adaptive_avg_pool2d_backward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyCoreOpsModel(nn.Module):
    """CNN that uses conv, group norm, max pool, avg pool, and adaptive avg pool."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        out_channels: int = 8,
        num_groups: int = 8,
    ):
        super().__init__()

        if hidden_channels % num_groups != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(num_groups, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(num_groups, hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through: Conv->GroupNorm->ReLU->MaxPool->Conv->
        GroupNorm->ReLU->AvgPool->AdaptiveAvgPool->Conv
        Output is always (batch, out_channels, 4, 4) regardless of
        input size.
        """
        x = F.relu(self.group_norm1(self.conv1(x)))
        x, _ = F.max_pool2d(x, kernel_size=2, return_indices=True)
        x = F.relu(self.group_norm2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=2)
        x = F.adaptive_avg_pool2d(x, output_size=(4, 4))
        x = self.conv_out(x)
        return x


def main():
    """Demonstrate the model with a forward/backward pass."""
    model = ToyCoreOpsModel(in_channels=3, hidden_channels=32, out_channels=8, num_groups=8)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 64, 64, requires_grad=True)

    model.train()
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    print("✓ Forward/backward pass completed")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Input: {input_tensor.shape} -> Output: {output.shape}")
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = len(list(model.parameters()))
    print(f"  Gradients computed: {grad_count}/{total_params}")
    return model


if __name__ == "__main__":
    main()
