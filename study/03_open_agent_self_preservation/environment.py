"""Capture the local software environment without credentials or machine paths."""

from __future__ import annotations

import os
import platform
import sys

import numpy
import peft
import safetensors
import torch
import transformers
from study3 import HERE, write_json

write_json(HERE / "environment.json", {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "logical_processors": os.cpu_count(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "numpy": numpy.__version__,
    "safetensors": safetensors.__version__,
})
