import argparse
import os
from model.cnn import SimpleCNN, ResNet32
from utilities.train_utils import (
    load_cifar10,
    train,
    save_checkpoint,
    load_checkpoint,
    LearningRateScheduler,
    DynamicDropout,
)


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 (NumPy/CuPy)")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "resnet32"],
        help="Model type: cnn (SimpleCNN) or resnet32 (ResNet-32)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/cifar-10-batches-py",
        help="CIFAR-10 data directory",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (auto-set if None)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["sgd", "adam"],
        help="Optimizer: sgd or adam (auto-set if None)",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="ADAM beta1 parameter (default: 0.9)"
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="ADAM beta2 parameter (default: 0.999)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=None,
        help="Dropout rate (auto-set if None)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoint",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Learning rate scheduling options
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear", "step"],
        help="Learning rate schedule type",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of warmup epochs (default: auto-set based on model)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for scheduling",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.1,
        help="Learning rate decay factor for step scheduling",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        nargs="*",
        default=None,
        help="Epochs at which to decay learning rate (for step scheduling)",
    )

    # Dynamic dropout options
    parser.add_argument(
        "--dropout-schedule",
        type=str,
        default="none",
        choices=["none", "linear_decay", "cosine_decay", "adaptive"],
        help="Dropout scheduling strategy",
    )
    parser.add_argument(
        "--min-dropout",
        type=float,
        default=0.1,
        help="Minimum dropout rate for dynamic scheduling",
    )
    parser.add_argument(
        "--max-dropout",
        type=float,
        default=0.7,
        help="Maximum dropout rate for dynamic scheduling",
    )

    args = parser.parse_args()

    print(f"[INFO] Loading CIFAR-10 from {args.data} ...")
    X_train, y_train, X_test, y_test = load_cifar10(args.data)
    print(f"[INFO] Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Set model-specific defaults
    if args.model == "cnn":
        # SimpleCNN optimized defaults for 200 epochs
        default_lr = 0.015  # Higher initial LR for longer training with cosine decay
        default_optimizer = "sgd"  # SGD works well with momentum for longer training
        default_dropout = 0.6  # Higher initial dropout for longer training
        default_epochs = (
            40 if args.epochs == 10 else args.epochs
        )  # 200 epochs for thorough training
        default_warmup = 6  # Longer warmup for higher learning rate
    elif args.model == "resnet32":
        # ResNet-32 optimized defaults
        default_lr = 0.01
        default_optimizer = "adam"
        default_dropout = 0.3
        default_epochs = 50 if args.epochs == 10 else args.epochs
        default_warmup = 5  # Longer warmup for ResNet
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Apply defaults if not explicitly set
    if args.lr is None:
        args.lr = default_lr
        print(f"[INFO] Using default learning rate: {args.lr}")

    if args.optimizer is None:
        args.optimizer = default_optimizer
        print(f"[INFO] Using default optimizer: {args.optimizer.upper()}")

    if args.dropout_rate is None:
        args.dropout_rate = default_dropout
        print(f"[INFO] Using default dropout rate: {args.dropout_rate}")

    if args.epochs == 10:  # User didn't change default
        args.epochs = default_epochs
        print(f"[INFO] Using recommended epochs for {args.model}: {args.epochs}")

    if args.warmup_epochs == 0:  # User didn't set warmup
        args.warmup_epochs = default_warmup
        print(f"[INFO] Using default warmup epochs: {args.warmup_epochs}")

    # Model-specific optimizations for longer training
    if args.model == "cnn":
        # Enable dynamic dropout by default for 200-epoch training
        if args.dropout_schedule == "none" and args.epochs >= 100:
            args.dropout_schedule = "linear_decay"
            print("[INFO] Enabling linear_decay dropout schedule for long training")

        # Set optimal min/max dropout for CNN
        if args.dropout_schedule != "none":
            if args.min_dropout == 0.1:  # Default value
                args.min_dropout = 0.2  # Higher min for CNN
                print(f"[INFO] Using optimized min dropout for CNN: {args.min_dropout}")
            if args.max_dropout == 0.7:  # Default value
                args.max_dropout = 0.7  # Keep max as is

        # Optimize step decay for CNN if using step schedule
        if args.lr_schedule == "step" and args.lr_decay_steps is None:
            # More frequent decay steps for 200 epochs
            args.lr_decay_steps = [50, 100, 150, 175]
            print(f"[INFO] Using optimized decay steps for CNN: {args.lr_decay_steps}")

    # Additional ResNet-specific adjustments
    if args.model == "resnet32":
        # Adjust batch size for better ResNet training if using default
        if args.batch_size == 64:  # Default value
            args.batch_size = 128
            print(f"[INFO] Using larger batch size for ResNet: {args.batch_size}")

        # Set default decay steps for step scheduling
        if args.lr_schedule == "step" and args.lr_decay_steps is None:
            args.lr_decay_steps = [args.epochs // 3, 2 * args.epochs // 3]
            print(f"[INFO] Using default decay steps: {args.lr_decay_steps}")

    # Initialize learning rate scheduler
    lr_scheduler = None
    if args.lr_schedule != "none":
        lr_scheduler = LearningRateScheduler(
            base_lr=args.lr,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            decay_type=args.lr_schedule,
            min_lr=args.min_lr,
            decay_factor=args.lr_decay_factor,
            decay_steps=args.lr_decay_steps,
        )
        print(
            f"[INFO] Using {args.lr_schedule} learning rate schedule with {args.warmup_epochs} warmup epochs"
        )

    # Initialize dynamic dropout scheduler
    dropout_scheduler = None
    if args.dropout_schedule != "none":
        dropout_scheduler = DynamicDropout(
            initial_rate=args.dropout_rate,
            min_rate=args.min_dropout,
            max_rate=args.max_dropout,
            strategy=args.dropout_schedule,
        )
        print(f"[INFO] Using {args.dropout_schedule} dropout schedule")

    if args.model == "cnn":
        print(f"[INFO] Using SimpleCNN model with {args.optimizer.upper()} optimizer")
        print(
            f"[INFO] Hyperparameters: lr={args.lr}, dropout={args.dropout_rate}, epochs={args.epochs}"
        )
        if args.optimizer.lower() == "adam":
            print(f"[INFO] ADAM parameters: beta1={args.beta1}, beta2={args.beta2}")
        model = SimpleCNN(
            num_classes=10,
            dropout_rate=args.dropout_rate,
            optimizer=args.optimizer,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
        )
    elif args.model == "resnet32":
        print(f"[INFO] Using ResNet-32 model with {args.optimizer.upper()} optimizer")
        print(
            f"[INFO] Hyperparameters: lr={args.lr}, dropout={args.dropout_rate}, epochs={args.epochs}"
        )
        if args.optimizer.lower() == "adam":
            print(f"[INFO] ADAM parameters: beta1={args.beta1}, beta2={args.beta2}")
        model = ResNet32(
            num_classes=10,
            dropout_rate=args.dropout_rate,
            optimizer=args.optimizer,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Store total epochs in model for dropout scheduler
    model._total_epochs = args.epochs

    # Calculate and display parameter count
    total_params = 0
    if hasattr(model, "blocks"):
        # ResNet model
        for attr_name in dir(model):
            if not attr_name.startswith("_") and attr_name != "blocks":
                attr_value = getattr(model, attr_name)
                if hasattr(attr_value, "size"):
                    total_params += attr_value.size

        for block in model.blocks:
            for attr_name in dir(block):
                if not attr_name.startswith("_"):
                    attr_value = getattr(block, attr_name)
                    if hasattr(attr_value, "size"):
                        total_params += attr_value.size
    else:
        # SimpleCNN model
        for attr_name in dir(model):
            if not attr_name.startswith("_"):
                attr_value = getattr(model, attr_name)
                if hasattr(attr_value, "size"):
                    total_params += attr_value.size

    print(f"[INFO] Model: {model.name}, Parameters: {total_params:,}")

    best_acc = 0.0
    best_path = None
    start_epoch = 0

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume:
        print(f"[INFO] Loading checkpoint from {args.resume}")
        resumed_acc = load_checkpoint(model, args.resume)
        if resumed_acc is not None:
            best_acc = resumed_acc
            print(f"[INFO] Resumed with best accuracy: {best_acc:.2f}%")
        # Extract epoch number from checkpoint filename if possible
        try:
            import re

            match = re.search(r"epoch(\d+)", args.resume)
            if match:
                start_epoch = int(match.group(1))
                print(f"[INFO] Resuming from epoch {start_epoch}")
        except (ValueError, AttributeError) as e:
            print(f"[WARNING] Could not extract epoch from checkpoint filename, {e}")
            pass

    for epoch in range(start_epoch, args.epochs):
        print(f"[INFO] Starting epoch {epoch + 1}/{args.epochs}")
        acc = train(
            model,
            X_train,
            y_train,
            args.batch_size,
            args.lr,
            _epoch=epoch,
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model,
            X_test=X_test,
            y_test=y_test,
            lr_scheduler=lr_scheduler,
            dropout_scheduler=dropout_scheduler,
        )
        if acc > best_acc:
            best_acc = acc
            best_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pkl")
            print(f"[DEBUG] Checkpoint path: {best_path}")
            save_checkpoint(model, best_path, accuracy=best_acc)
            print(f"[INFO] New best model saved: {best_path}")

    print(f"[RESULT] Best test accuracy: {best_acc:.2f}% (model: {best_path})")


if __name__ == "__main__":
    main()
