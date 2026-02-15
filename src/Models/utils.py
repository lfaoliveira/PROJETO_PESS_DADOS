import numpy as np
import pandas as pd
import torch
from typing import cast

from DataProcesser.dataset import StrokeDataset


def analyse_test(
    model: torch.nn.Module,
    test_subset: torch.utils.data.Subset[StrokeDataset],
    output_df: pd.DataFrame,
):
    """
    Performs a one-pass analysis logic on the GPU.
    """
    device = next(model.parameters()).device
    model.eval()

    # Move data to GPU for fast inference
    # Note: data and labels are assumed to be Tensors in the underlying dataset
    indices = torch.tensor(test_subset.indices, device=device)

    dataset = cast(StrokeDataset, test_subset.dataset)
    data = dataset.data[indices].to(device)
    labels = dataset.labels[indices].to(device).long().squeeze()

    with torch.no_grad():
        logits = model(data)
        preds = torch.argmax(logits, dim=1)

    # 1. Compute result codes on GPU (TP=1, FP=2, FN=3, TN=4)
    results_code = torch.zeros_like(preds, dtype=torch.uint8)
    results_code[(preds == 1) & (labels == 1)] = 1
    results_code[(preds == 1) & (labels == 0)] = 2
    results_code[(preds == 0) & (labels == 1)] = 3
    results_code[(preds == 0) & (labels == 0)] = 4

    # 2. Convert to CPU for DataFrame assignment
    preds_np = preds.cpu().numpy()
    codes_np = results_code.cpu().numpy()

    # Efficiently map codes to labels
    code_map = {1: "TP", 2: "FP", 3: "FN", 4: "TN", 0: "ERROR"}
    results_str = np.vectorize(code_map.get)(codes_np)

    # 3. Batch assignment to the DataFrame
    target_indices = list(test_subset.indices)
    output_df.loc[target_indices, "pred"] = preds_np
    output_df.loc[target_indices, "error"] = results_str

    return output_df, logits, labels
