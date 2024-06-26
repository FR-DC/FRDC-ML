from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from frdc.train.stratified_sampling import RandomStratifiedSampler


def test_stratifed_sampling_has_correct_probs():
    sampler = RandomStratifiedSampler(["A", "A", "B"])

    assert torch.all(sampler.target_probs == torch.tensor([0.25, 0.25, 0.5]))


def test_stratified_sampling_fairly_samples():
    """This test checks that the stratified sampler works with a dataloader."""

    # This is a simple example of a dataset with 2 classes.
    # The first 2 samples are class 0, the third is class 1.
    x = torch.tensor([0, 1, 2])
    y = ["A", "A", "B"]

    # To check that it's truly stratified, we'll sample 1000 times
    # then assert that both classes are sampled roughly equally.

    # In this case, the first 2 x should be sampled roughly 250 times,
    # and the third x should be sampled roughly 500 times.

    num_samples = 1000
    batch_size = 10
    dl = DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        sampler=RandomStratifiedSampler(y, num_samples=num_samples),
    )

    # Note that when we sample from a TensorDataset, we get a tuple of tensors.
    # So we need to unpack the tuple.
    x_samples = torch.cat([x for (x,) in dl])

    assert len(x_samples) == num_samples
    assert torch.allclose(
        torch.bincount(x_samples),
        torch.tensor([250, 250, 500]),
        # atol is the absolute tolerance, so the result can differ by 50
        atol=50,
    )
