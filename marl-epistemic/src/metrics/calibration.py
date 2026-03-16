"""Calibration metrics: Expected Calibration Error and reliability diagrams."""

import torch
from torch import Tensor


def expected_calibration_error(
    y_true: Tensor,
    y_pred_prob: Tensor,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where B_b is the set of predictions falling in bin b,
    acc is the accuracy in that bin, and conf is the mean confidence.

    Args:
        y_true: Binary labels. Shape: (N,).
        y_pred_prob: Predicted probabilities. Shape: (N,).
        n_bins: Number of calibration bins.

    Returns:
        ECE value in [0, 1].
    """
    y_true = y_true.float()
    y_pred_prob = y_pred_prob.float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = y_true.shape[0]

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_pred_prob > lo) & (y_pred_prob <= hi)
        if i == 0:
            mask = mask | (y_pred_prob == lo)

        n_in_bin = mask.sum().item()
        if n_in_bin == 0:
            continue

        accuracy = y_true[mask].mean().item()
        confidence = y_pred_prob[mask].mean().item()
        ece += (n_in_bin / total) * abs(accuracy - confidence)

    return ece


def reliability_diagram_data(
    y_true: Tensor,
    y_pred_prob: Tensor,
    n_bins: int = 10,
) -> dict:
    """
    Compute data for reliability diagram.

    Returns dict with:
        bin_centers: (n_bins,) centers of each bin
        bin_accuracies: (n_bins,) empirical accuracy in each bin
        bin_confidences: (n_bins,) mean predicted probability in each bin
        bin_counts: (n_bins,) number of samples in each bin
    """
    y_true = y_true.float()
    y_pred_prob = y_pred_prob.float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    centers = []
    accuracies = []
    confidences = []
    counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_pred_prob > lo) & (y_pred_prob <= hi)
        if i == 0:
            mask = mask | (y_pred_prob == lo)

        n_in_bin = mask.sum().item()
        centers.append((lo.item() + hi.item()) / 2)
        counts.append(n_in_bin)

        if n_in_bin == 0:
            accuracies.append(0.0)
            confidences.append((lo.item() + hi.item()) / 2)
        else:
            accuracies.append(y_true[mask].mean().item())
            confidences.append(y_pred_prob[mask].mean().item())

    return {
        "bin_centers": centers,
        "bin_accuracies": accuracies,
        "bin_confidences": confidences,
        "bin_counts": counts,
    }


def brier_score(y_true: Tensor, y_pred_prob: Tensor) -> float:
    """Brier score: mean squared error of probability predictions."""
    return (y_pred_prob.float() - y_true.float()).pow(2).mean().item()
