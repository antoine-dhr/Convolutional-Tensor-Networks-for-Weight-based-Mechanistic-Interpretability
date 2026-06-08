# Convolutional tensor networks for weight-based mechanistic interpretability (master's thesis) — Code

This repository contains the code accompanying the master's thesis. It is organised into two
separate projects, each with its own README, dependencies, and usage instructions.

- **[`src/cnn_to_tn/`](cnn_to_tn/README.md) - Convolutional Neural Networks to Tensor Networks.**
  Contains the code to build, train, and convert the newly proposed CNN architceure into an equivalent tensor network. The proposed architecture is a multi-stage deep residual CNN consisting entirely from components that are individually convertible to a tensor network component equivalent. This architecture was trained and evaluated on MNIST, CIFAR-10, CIFAR-100, and SVHN.

- **[`src/bilinear_decomposition/`](bilinear_decomposition/README.md) - Weight-based bilinear convolutional model analysis.**
  Contains the code to perform a weight-based analysis on a bilinear MLP and a bilinear convolutional model. It extends the work of [tdooms/bilinear-decomposition](https://github.com/tdooms/bilinear-decomposition).

## Repository structure

```
src/
├── cnn_to_tn/              # CNN architecture definition + tensor-network conversion
└── bilinear_decomposition/ # Weight-based bilinear convolutional analysis
```

## Getting started

Each subproject is managed independently with [uv](https://docs.astral.sh/uv/) and has its own
`pyproject.toml`. 
