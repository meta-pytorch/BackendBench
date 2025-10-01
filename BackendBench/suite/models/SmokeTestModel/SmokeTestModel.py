# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Smoke test model focused on matrix multiplication operations.

This model is designed to test mm operations in both forward and backward passes.
It uses explicit torch.mm calls to ensure matrix multiplication operations are triggered.

The model implements a simple architecture with:
1. Matrix multiplication operations
2. ReLU activations
3. Element-wise operations

Usage:
    python SmokeTestModel.py

This will create a model with default configuration and run a simple forward/backward pass
to demonstrate that mm operations are used.
"""

import torch
import torch.nn as nn


class SmokeTestModel(nn.Module):
    """
    Simple model focused on testing matrix multiplication operations.

    This model uses explicit torch.mm operations to ensure we trigger
    aten.mm.default in both forward and backward passes.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 64,
    ):
        """
        Initialize the SmokeTestModel.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super().__init__()

        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Weight matrices for explicit mm operations
        self.weight1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.weight2 = nn.Parameter(torch.randn(hidden_dim, output_dim))

        # Bias terms
        self.bias1 = nn.Parameter(torch.randn(hidden_dim))
        self.bias2 = nn.Parameter(torch.randn(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using explicit mm operations.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # First mm operation: x @ weight1
        # This triggers aten.mm.default in forward
        x = torch.mm(x, self.weight1)
        x = x + self.bias1
        x = torch.relu(x)

        # Second mm operation: x @ weight2
        # This triggers aten.mm.default again in forward
        x = torch.mm(x, self.weight2)
        x = x + self.bias2

        return x


def main():
    """
    Demonstrate the SmokeTestModel with a simple forward/backward pass.
    """
    print("SmokeTestModel Demonstration")
    print("=" * 50)

    # Create model with default configuration
    model = SmokeTestModel(
        input_dim=128,
        hidden_dim=256,
        output_dim=64,
    )

    # Create sample input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 128, requires_grad=True)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Input shape: {input_tensor.shape}")

    # Forward pass
    model.train()
    output = model(input_tensor)
    expected_shape = torch.Size([batch_size, 64])

    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {expected_shape}")
    print(f"Shape matches: {output.shape == expected_shape}")

    # Perform backward pass to trigger mm operations in backward
    print("\nPerforming backward pass...")
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass completed successfully")

    # Check gradients were computed
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = len(list(model.parameters()))
    print(f"✓ Gradients computed for {grad_count}/{total_params} parameters")

    print("\n✓ Model demonstration completed successfully!")
    print("This model is ready to be used with the Model Suite for testing mm operators.")

    return model


if __name__ == "__main__":
    main()
