"""Shared positive-edge entropy selector from the published training loop."""
import numpy as np
import torch


def select_batch_indices(positive_probabilities, rate):
    entropy = -(positive_probabilities * torch.log(positive_probabilities) +
                (1 - positive_probabilities) * torch.log(1 - positive_probabilities))
    if not torch.isfinite(entropy).all():
        raise ValueError("Non-finite selector entropy; probabilities are not clipped")
    _, indices = torch.topk(entropy, k=int(rate * len(positive_probabilities)))
    # Keep the entropy top-k set, restore the source event order for last-message aggregation.
    return np.sort(indices.detach().cpu().numpy())
