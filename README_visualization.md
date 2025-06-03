# Training Log Visualization

## Overview

This module provides visualization based on CNN CIFAR-10 train/test logs.

## Features

- Training/validation loss curves
- Accuracy progression over epochs
- Loss distribution analysis
- Overfitting detection
- Summary statistics

## Usage

### Basic Usage

```bash
# Configure common dependencies
uv sync --group viz

uv run visualize_logs.py \
  --train-log log/resnet32.log \
  --test-log log/resnet32_test.log \
  --prefix resnet32
```

## Output Files

- `model_training_progress.png` - Main training curves
- `model_comparison.png` - Train vs test comparison
- `model_per_class_accuracy.png` - Test per-class accuracy result
- `model_training_summary.txt` - Statistical summary

## Customization

Modify the regex patterns in `parse_log_file()` to match your specific log format.
