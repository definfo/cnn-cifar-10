# CNN CIFAR-10 Implementation

A pure NumPy/CuPy implementation of Convolutional Neural Networks for CIFAR-10 classification.

## Models

### SimpleCNN

- Basic CNN with convolution, batchnorm, ReLU, maxpooling, fully connected layers
- Lightweight architecture suitable for quick experiments
- **Recommended settings**: SGD optimizer, lr=0.01, dropout=0.5, 20 epochs

### ResNet-32

- 32-layer Residual Network implementation
- Uses residual blocks with skip connections
- Better performance on complex datasets
- **Recommended settings**: Adam optimizer, lr=0.01, dropout=0.3, 50+ epochs, batch_size=128

## Quick Start

### Preparation

Before diving in, ensure that you have Nix installed on your system. If not, you
can download and install it from the official
[Nix website](https://nixos.org/download.html) or from the
[Determinate Systems installer](https://github.com/DeterminateSystems/nix-installer).
If running on macOS, you need to have Nix-Darwin installed, as well. You can
follow the installation instruction on
[GitHub](https://github.com/LnL7/nix-darwin?tab=readme-ov-file#flakes).

You may also install [direnv](https://direnv.net/docs/installation.html) for better
devshell integration.

```bash
# with direnv (see `.envrc` for details)
direnv allow

# without direnv
nix develop .#impure

# CUDA
nix develop .#cuda
```

After entering devshell, you may configure Python deps through uv:

```bash
uv sync
```

### Training with Default Settings (Recommended)

```bash
# SimpleCNN with optimized defaults
uv run src/train_cli.py --model cnn

# ResNet-32 with optimized defaults
uv run src/train_cli.py --model resnet32
```

The framework automatically sets optimal hyperparameters for each model:

| Model     | Optimizer | Learning Rate | Dropout | Batch Size | Epochs |
| --------- | --------- | ------------- | ------- | ---------- | ------ |
| SimpleCNN | SGD       | 0.01          | 0.5     | 64         | 20     |
| ResNet-32 | Adam      | 0.01          | 0.3     | 128        | 50     |

### Custom Training

```bash
# Custom SimpleCNN
uv run src/train_cli.py \
    --model cnn \
    --lr 0.005 \
    --optimizer adam \
    --epochs 30

# Custom ResNet-32
uv run src/train_cli.py \
    --model resnet32 \
    --lr 0.0005 \
    --batch-size 256 \
    --dropout-rate 0.2 \
    --epochs 100
```

### Advanced Options

```bash
uv run src/train_cli.py \
    --model resnet32 \
    --lr 0.001 \
    --optimizer adam \
    --beta1 0.9 \
    --beta2 0.999 \
    --batch-size 128 \
    --dropout-rate 0.3 \
    --epochs 100 \
    --checkpoint-dir checkpoints \
    --data data/cifar-10-batches-py
```

### Resume Training

```bash
uv run src/train_cli.py --model resnet32 --resume checkpoints/resnet32_epoch25.pkl
```

## Performance Expectations

### SimpleCNN

- **Training time**: ~5-10 minutes (20 epochs, CPU)
- **Expected accuracy**: 65-75% on CIFAR-10
- **Parameters**: ~50K

### ResNet-32

- **Training time**: ~30-60 minutes (50 epochs, CPU)
- **Expected accuracy**: 80-90% on CIFAR-10
- **Parameters**: ~460K

## Model Architecture

### SimpleCNN

- Input: 3×32×32 (CIFAR-10 images)
- Conv2D: 3→8 channels, 3×3 kernel
- BatchNorm + ReLU + MaxPool (2×2)
- Flatten: 8×15×15 → 1800
- FC: 1800→128 + BatchNorm + ReLU + Dropout
- FC: 128→10 (classes)

### ResNet-32

- Input: 3×32×32 (CIFAR-10 images)
- Initial Conv: 3→16 channels, 3×3 kernel + BatchNorm + ReLU
- **Stage 1**: 5 residual blocks (16 channels, 32×32)
- **Stage 2**: 5 residual blocks (32 channels, 16×16, first block stride=2)
- **Stage 3**: 5 residual blocks (64 channels, 8×8, first block stride=2)
- Global Average Pooling: 64×8×8 → 64
- Dropout + FC: 64→10 (classes)

Each residual block contains:

- Conv 3×3 + BatchNorm + ReLU
- Conv 3×3 + BatchNorm
- Skip connection (with 1×1 conv for dimension matching if needed)
- Final ReLU
