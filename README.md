# CNN CIFAR-10 Implementation

A pure NumPy/CuPy implementation of Convolutional Neural Networks for CIFAR-10 classification.

## Models

### SimpleCNN

- Basic CNN with convolution, batchnorm, ReLU, maxpooling, fully connected layers
- Lightweight architecture suitable for thorough training
- **Recommended settings**:
  - SGD optimizer
  - lr=0.02
  - dropout=0.6→0.2 (linear decay), 200 epochs
  - cosine schedule with 10 warmup epochs

### ResNet-32

- 32-layer Residual Network implementation
- Uses residual blocks with skip connections
- Better performance on complex datasets
- **Recommended settings**:
  - Adam optimizer
  - lr=0.01
  - dropout=0.3, 50+ epochs
  - cosine schedule with 5 warmup epochs
  - batch_size=128

## Features

### Learning Rate Scheduling

- **Cosine Annealing**: Smooth decay with restarts capability
- **Linear Decay**: Simple linear reduction over training
- **Step Decay**: Discrete drops at specified epochs
- **Warmup**: Gradual increase from 0 to base learning rate

### Dynamic Dropout

- **Linear Decay**: Gradually reduce dropout during training
- **Cosine Decay**: Smooth dropout reduction following cosine curve
- **Adaptive**: Adjust dropout based on validation performance
- **Constant**: Traditional fixed dropout rate

## Quick Start

### Preparation

```bash
# Clone this repo
git clone https://github.com/definfo/cnn-cifar-10.git
cd cnn-cifar-10

# (Optional) Reuse pre-trained model checkpoints
git lfs pull

# Install Nix
curl -fsSL https://install.determinate.systems/nix \
  | sh -s -- install --determinate

# If direnv is installed, (see `.envrc` for details)
# The following command can automatically load the
# devshell environment when cd into this directory.

# direnv allow

# Otherwise, enter the devshell with one of the
# following commands.
# NumPy
nix develop .#impure

# OR CuPy with CUDA Backend
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
# SimpleCNN with optimized defaults (200 epochs, cosine schedule + warmup + dynamic dropout)
uv run src/train_cli.py --model cnn

# ResNet-32 with optimized defaults (cosine schedule + warmup)
uv run src/train_cli.py --model resnet32

# OR resume training on pre-trained weights
git lfs pull
uv run src/train_cli.py --model cnn --resume checkpoint/cnn_best.pkl.bak
```

The framework automatically sets optimal hyperparameters for each model:

| Model     | Optimizer | Base LR | Schedule | Warmup | Dropout         | Batch Size | Epochs |
| --------- | --------- | ------- | -------- | ------ | --------------- | ---------- | ------ |
| SimpleCNN | SGD       | 0.02    | Cosine   | 10     | 0.6→0.2 (decay) | 64         | 200    |
| ResNet-32 | Adam      | 0.01    | Cosine   | 5      | 0.3             | 128        | 50     |

### Custom Training

```bash
# Custom SimpleCNN with step scheduling (optimized for 200 epochs)
uv run src/train_cli.py \
    --model cnn \
    --lr 0.02 \
    --lr-schedule step \
    --lr-decay-steps 50 100 150 175 \
    --lr-decay-factor 0.1 \
    --dropout-schedule linear_decay \
    --epochs 200

# Custom ResNet-32 with adaptive dropout
uv run src/train_cli.py \
    --model resnet32 \
    --lr 0.01 \
    --lr-schedule cosine \
    --warmup-epochs 10 \
    --dropout-schedule adaptive \
    --min-dropout 0.1 \
    --max-dropout 0.5 \
    --epochs 100
```

### Advanced Training Options

```bash
# SimpleCNN with full control over 200-epoch training
uv run src/train_cli.py \
    --model cnn \
    --lr 0.025 \
    --lr-schedule cosine \
    --warmup-epochs 15 \
    --min-lr 1e-6 \
    --optimizer sgd \
    --dropout-rate 0.6 \
    --dropout-schedule linear_decay \
    --min-dropout 0.2 \
    --max-dropout 0.7 \
    --batch-size 64 \
    --epochs 200 \
    --checkpoint-dir checkpoint \
    --data data/cifar-10-batches-py
```

### Learning Rate Scheduling Options

- `--lr-schedule`: `none`, `cosine`, `linear`, `step`
- `--warmup-epochs`: Number of warmup epochs (default: 2 for CNN, 5 for ResNet)
- `--min-lr`: Minimum learning rate (default: 1e-6)
- `--lr-decay-factor`: Decay factor for step scheduling (default: 0.1)
- `--lr-decay-steps`: Epochs for step decay (default: auto-calculated)

### Dynamic Dropout Options

- `--dropout-schedule`: `none`, `linear_decay`, `cosine_decay`, `adaptive`
- `--min-dropout`: Minimum dropout rate (default: 0.1)
- `--max-dropout`: Maximum dropout rate (default: 0.7)

### Run test on checkpoint

```bash
uv run src/test_cli.py --model resnet32 --resume checkpoint/resnet32_best.pkl
```

## Performance Expectations

### SimpleCNN

- **Training time**:
  - 66 min total (Intel Core i7-12700H CPU, 200 epochs)
  - 20 min total (NVIDIA GeForce RTX 3060 Laptop GPU, 200 epochs)
- **Expected accuracy**: 70-80% on CIFAR-10 (with 200-epoch training + scheduling)
- **Parameters**: ~50K

### ResNet-32

- **Training time**:
  - 50 min / epoch (Intel Core i7-12700H CPU)
  - 7.5 min / epoch (NVIDIA GeForce RTX 3060 Laptop GPU)
- **Expected accuracy**: 75-80% on CIFAR-10 (with scheduling)
- **Parameters**: ~460K

## Scheduling Strategies

### Learning Rate Scheduling

1. **Cosine Annealing** (Recommended): Provides smooth decay with good convergence properties
2. **Linear Decay**: Simple and effective for shorter training runs
3. **Step Decay**: Good for fine-tuning and when you know optimal decay points
   - **SimpleCNN 200-epoch optimized**: Steps at epochs 50, 100, 150, 175
4. **Warmup**: Essential for training stability, especially with higher learning rates

### Dynamic Dropout

1. **Linear Decay**: Gradually reduce overfitting as model learns (recommended for 200-epoch training)
2. **Cosine Decay**: Smooth dropout reduction following cosine curve
3. **Adaptive**: Automatically adjust based on validation performance
4. **Constant**: Traditional approach, good baseline

### Training Strategy for SimpleCNN (200 epochs)

The 200-epoch training strategy for SimpleCNN is designed for maximum performance:

1. **High initial learning rate (0.02)** with cosine decay to explore the parameter space effectively
2. **Longer warmup (10 epochs)** to stabilize training with the higher learning rate
3. **Dynamic dropout (0.6→0.2)** to prevent overfitting early while allowing the model to learn complex patterns later
4. **Extended training (200 epochs)** to fully converge and achieve optimal performance

This strategy typically achieves 10-15% higher accuracy than the previous 20-epoch default.

## Visualization

See [README_visualization.md](./README_visualization.md)

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

- Test accuracy during training appears to be lower than actual value.

- Current model serde implementation is not optimal and may contain garbage variables.

- `train_utils.py` contains fragile vibe-coding snippets, which requires refactoring.
