# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

def fastp(correctness, performance, p):
    assert len(correctness) == len(performance), "correctness and performance must have the same length"
    correctness, performance = torch.tensor(correctness).bool(), torch.tensor(performance)
    return torch.where(correctness, performance > p, 0).float().mean()
