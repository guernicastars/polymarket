"""Tests for Keynesian loss function."""

import pytest
import torch

from src.learning.keynesian_loss import keynesian_loss, KeynesianLossFunction


class TestKeynesianLoss:
    def test_basic_computation(self):
        y_true = torch.tensor([[0.5], [0.8]])
        y_pred = torch.tensor([[0.4], [0.7]])
        evidence = torch.tensor([10.0, 5.0])

        loss = keynesian_loss(y_true, y_pred, evidence, mu=0.1)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_zero_mu_equals_mse(self):
        y_true = torch.randn(32, 1)
        y_pred = torch.randn(32, 1)
        evidence = torch.ones(32)

        loss_k = keynesian_loss(y_true, y_pred, evidence, mu=0.0)
        loss_mse = torch.nn.functional.mse_loss(y_pred, y_true)

        assert torch.allclose(loss_k, loss_mse, atol=1e-5)

    def test_higher_evidence_lower_penalty(self):
        y_true = torch.tensor([[0.5]])
        y_pred = torch.tensor([[0.5]])  # perfect prediction

        loss_low_ev = keynesian_loss(y_true, y_pred, torch.tensor([1.0]), mu=0.1)
        loss_high_ev = keynesian_loss(y_true, y_pred, torch.tensor([100.0]), mu=0.1)

        assert loss_low_ev > loss_high_ev

    def test_gradient_flows(self):
        y_true = torch.tensor([[0.5]], requires_grad=False)
        y_pred = torch.tensor([[0.4]], requires_grad=True)
        evidence = torch.tensor([5.0], requires_grad=True)

        loss = keynesian_loss(y_true, y_pred, evidence, mu=0.1)
        loss.backward()

        assert y_pred.grad is not None
        assert evidence.grad is not None

    def test_bce_mode(self):
        y_true = torch.tensor([[1.0], [0.0]])
        y_pred = torch.tensor([[0.5], [-0.5]])  # logits
        evidence = torch.tensor([5.0, 5.0])

        loss = keynesian_loss(y_true, y_pred, evidence, mu=0.1, base_loss="bce")
        assert loss.item() > 0


class TestKeynesianLossFunction:
    def test_warmup_schedule(self):
        loss_fn = KeynesianLossFunction(mu=1.0, mu_schedule="warmup", mu_warmup_steps=100)
        assert loss_fn.get_mu() == 0.0  # step 0

        # Simulate steps
        for _ in range(50):
            loss_fn._step += 1
        assert 0.4 < loss_fn.get_mu() < 0.6

        for _ in range(50):
            loss_fn._step += 1
        assert loss_fn.get_mu() == 1.0

    def test_callable(self):
        loss_fn = KeynesianLossFunction(mu=0.1)
        y_true = torch.randn(16, 1)
        y_pred = torch.randn(16, 1)
        evidence = torch.ones(16) * 5.0

        loss = loss_fn(y_true, y_pred, evidence)
        assert loss.dim() == 0

    def test_reset(self):
        loss_fn = KeynesianLossFunction(mu=1.0, mu_schedule="warmup", mu_warmup_steps=10)
        for _ in range(10):
            loss_fn._step += 1
        assert loss_fn._step == 10
        loss_fn.reset()
        assert loss_fn._step == 0
