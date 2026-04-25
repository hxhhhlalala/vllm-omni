# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .wan2_2 import (
    Wan22I2VForVideoGeneration,
    Wan22T2VForVideoGeneration,
    Wan22TI2VForVideoGeneration,
)

__all__ = [
    "Wan22T2VForVideoGeneration",
    "Wan22I2VForVideoGeneration",
    "Wan22TI2VForVideoGeneration",
]
