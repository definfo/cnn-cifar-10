import argparse
import os
from model.cnn import SimpleCNN, ResNet32
from utilities.train_utils import (
    load_cifar10,
    train,
    save_checkpoint,
    load_checkpoint,
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
    args = parser.parse_args()

    print(f"[INFO] Loading CIFAR-10 from {args.data} ...")
    X_train, y_train, X_test, y_test = load_cifar10(args.data)
    print(f"[INFO] Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Set model-specific defaults
    if args.model == "cnn":
        # SimpleCNN defaults
        default_lr = 0.01
        default_optimizer = "sgd"
        default_dropout = 0.5
        default_epochs = (
            20 if args.epochs == 10 else args.epochs
        )  # Increase default epochs
    elif args.model == "resnet32":
        # ResNet-32 optimized defaults
        # Use 0.01 for ResNet-32 - it's deeper but still relatively small (32 layers)
        # 0.001 is too conservative for this size network on CIFAR-10
        default_lr = 0.01  # Same as SimpleCNN but with Adam's adaptive rates
        default_optimizer = (
            "adam"  # Adam works better for ResNet with good learning rates
        )
        default_dropout = 0.3  # Lower dropout for deeper networks
        default_epochs = 50 if args.epochs == 10 else args.epochs  # More epochs needed

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

    # Additional ResNet-specific adjustments
    if args.model == "resnet32":
        # Adjust batch size for better ResNet training if using default
        if args.batch_size == 64:  # Default value
            args.batch_size = 128
            print(f"[INFO] Using larger batch size for ResNet: {args.batch_size}")

        # Adjust Adam parameters for ResNet
        if args.optimizer.lower() == "adam":
            # Use slightly different beta values for ResNet - more aggressive learning
            if args.beta1 == 0.9 and args.beta2 == 0.999:  # Default values
                args.beta1 = 0.9  # Keep standard first moment decay
                args.beta2 = 0.999  # Keep standard second moment decay
                print(
                    f"[INFO] Using ADAM parameters: beta1={args.beta1}, beta2={args.beta2}"
                )

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
            X_test=X_test,
            y_test=y_test,
        )
        if acc > best_acc:
            best_acc = acc
            best_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pkl")
            save_checkpoint(model, best_path, accuracy=best_acc)
            print(f"[INFO] New best model saved: {best_path}")

    print(f"[RESULT] Best test accuracy: {best_acc:.2f}% (model: {best_path})")


if __name__ == "__main__":
    main()
