from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray


TensorInput: TypeAlias = pd.DataFrame | NDArray[np.float64] | NDArray[np.float32]


def _to_tensor(data: TensorInput) -> torch.Tensor:
    """Convert tabular data into a float32 PyTorch tensor."""
    if isinstance(data, pd.DataFrame):
        array = data.to_numpy(dtype=np.float32, copy=True)
    else:
        array = np.asarray(data, dtype=np.float32)

    return torch.from_numpy(array.copy())