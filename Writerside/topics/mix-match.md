# MixMatch

In FRDC-ML, we leverage semi-supervised learning to improve the model's
performance through better augmentation consistency and using even unlabelled
data.

The algorithm we use is [MixMatch](https://arxiv.org/abs/1905.02249).
A state-of-the-art semi-supervised learning algorithm.
It is based on the idea of consistency regularization, which
encourages models to predict the same class even after augmentations that
occur naturally in the real world.

> In other words, a picture of a dog should be classified as a dog even if it
> is horizontally flipped, offset, or is of a different size. Consistency
> regularization encourages the model to predict the same class consistently.

Our implementation of MixMatch is a refactored version of
[YU1ut/MixMatch-pytorch](https://github.com/YU1ut/MixMatch-pytorch/tree/master)
We've refactored the code to follow more modern PyTorch practices, allowing us
to utilize it with modern PyTorch frameworks such as PyTorch Lightning.

We won't go through the details of MixMatch here, see
[Our Documentation](https://fr-dc.github.io/MixMatch-PyTorch-CIFAR10/pipeline.html)
in
our [MixMatch-PyTorch-CIFAR10](https://github.com/FR-DC/MixMatch-PyTorch-CIFAR10)
repository for more details.

## Implementation Details

1) How we implemented the MixMatch logic 
   [MixMatchModule](mix-match-module.md)

2) How we implemented the unique MixMatch data loading logic
   [Custom MixMatch Data Loading](custom-k-aug-dataloaders.md)

## References

- [YU1ut's PyTorch Implementation](https://github.com/YU1ut/MixMatch-pytorch/tree/master)
- [Google Research's TensorFlow Implementation](https://github.com/google-research/mixmatch)
