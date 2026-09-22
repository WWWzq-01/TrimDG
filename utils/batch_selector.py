"""Shared positive-edge entropy selector from the published training loop."""
import torch


def select_batch_indices(positive_probabilities, rate):
    entropy = -(positive_probabilities * torch.log(positive_probabilities) +
                (1 - positive_probabilities) * torch.log(1 - positive_probabilities))
    if not torch.isfinite(entropy).all():
        raise ValueError("Non-finite selector entropy; probabilities are not clipped")
    _, indices = torch.topk(entropy, k=int(rate * len(positive_probabilities)))
    return indices.detach().cpu().numpy()
