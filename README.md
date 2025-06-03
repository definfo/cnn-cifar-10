# CNN CIFAR-10 Implementation

A pure NumPy/CuPy implementation of Convolutional Neural Networks for CIFAR-10 classification.

## Models

### SimpleCNN

- Basic CNN with convolution, batchnorm, ReLU, maxpooling, fully connected layers
- Lightweight architecture suitable for quick experiments
- **Recommended settings**: SGD optimizer, lr=0.01, dropout=0.5, 200 epochs

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
# Clone this repo
git clone https://github.com/definfo/cnn-cifar-10.git
cd cnn-cifar-10

# (Optional) Reuse pre-trained model checkpoints
# NOTE: this checkpoint is only usable with CuPy backend
git lfs pull

# with direnv (see `.envrc` for details)
direnv allow

# without direnv
nix develop .#impure

# CUDA
nix develop .#cuda
```

After entering devshell, you may configure Python deps through uv:

```bash
# Configure common dependencies
uv sync

### Following steps are CUDA-only ###
# Configure CUDA dependencies
uv sync --group cuda

# (Optional) Configure CuPy with CUDA libraries for extra speedup
# `<library>` can be replaced by cutensor/nccl/cudnn
uv run python -m cupyx.tools.install_library --library <library> --cuda 12.x

# Check CuPy status
uv run python -c "import cupy; import cupy.cuda.cudnn; import cupy.cuda.nccl; cupy.show_config()"
```

Other possible dependencies:

- Linux kernel: `6.12.30 (mainline)`

- NVIDIA driver: `nvidia_x11-open 570.153.02`

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
    --checkpoint-dir checkpoint \
    --data data/cifar-10-batches-py
```

### Resume Training

```bash
uv run src/train_cli.py --model resnet32 --resume checkpoint/resnet32_epoch25.pkl
```

### Run test on checkpoint

```bash
uv run src/test_cli.py --model resnet32 --resume checkpoint/resnet32_best.pkl
```

## Performance Expectations

### SimpleCNN

- **Training time**:
  - 20 sec / epoch (Intel Core i7-12700H CPU)
  - 6 sec / epoch (NVIDIA GeForce RTX 3060 Laptop GPU)
- **Expected accuracy**: 55-65% on CIFAR-10
- **Parameters**: ~50K

### ResNet-32

- **Training time**:
  - 50 min / epoch (Intel Core i7-12700H CPU)
  - 7.5 min / epoch (NVIDIA GeForce RTX 3060 Laptop GPU)
- **Expected accuracy**: 70-75% on CIFAR-10
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

## Known issues

1. Model checkpoints cannot be reused across backends (NumPy/CuPy).

1. Test accuracy during training appears to be lower than actual value.

1. Current model serde implementation is not optimal and may contain garbage variables.

1. `train_utils.py` contains fragile vibe-coding snippets, which requires refactoring.
