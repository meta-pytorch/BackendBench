# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple model that tests matrix multiplication operations using explicit
torch.mm calls.
"""

import torch
import torch.nn as nn


class SmokeTestModel(nn.Module):
    """
    Model that uses explicit torch.mm operations to test aten.mm.default
    in forward/backward.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.weight1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.weight2 = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.bias1 = nn.Parameter(torch.randn(hidden_dim))
        self.bias2 = nn.Parameter(torch.randn(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (x @ weight1 + bias1) -> relu -> (x @ weight2 + bias2)
        """
        x = torch.mm(x, self.weight1) + self.bias1
        x = torch.relu(x)
        x = torch.mm(x, self.weight2) + self.bias2
        return x


def main():
    """Demonstrate the model with a forward/backward pass."""
    model = SmokeTestModel(input_dim=128, hidden_dim=128, output_dim=128)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 128, requires_grad=True)

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


if __name__ == "__main__":
    main()
