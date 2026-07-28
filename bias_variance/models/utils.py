import numpy as np
import pandas as pd
import torch


def _to_tensor(dataframe: pd.DataFrame) -> torch.Tensor:
    array = dataframe.to_numpy(
        dtype=np.float32,
        copy=True
    )

    return torch.from_numpy(array)