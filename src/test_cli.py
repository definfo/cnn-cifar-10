import argparse
import os
import json
from datetime import datetime
from model.cnn import SimpleCNN, ResNet32
from utilities.train_utils import load_cifar10, load_checkpoint


def evaluate_model(model, X_test, y_test, batch_size=128, verbose=True):
    """Evaluate model on test dataset"""
    from utilities.backend import xp, to_cpu

    # Set model to evaluation mode
    model.eval()

    num_samples = X_test.shape[0]
    correct = 0
    total = 0

    if verbose:
        print(f"Evaluating on {num_samples} test samples...")

    # Batch processing like in test() function
    for i in range(0, num_samples, batch_size):
        X_batch = X_test[i : i + batch_size]
        y_batch = y_test[i : i + batch_size]

        # Forward pass
        out = model.forward(X_batch)
        pred = xp.argmax(out, axis=1)

        # Accumulate results
        correct += int(xp.sum(pred == y_batch))
        total += y_batch.shape[0]

    # Calculate overall accuracy
    accuracy = 100.0 * correct / total

    if verbose:
        print(f"Overall Test Accuracy: {accuracy:.2f}%")

    # For detailed analysis, make another pass for per-class accuracy
    class_accuracies = {}
    if verbose:
        print("Calculating per-class accuracy...")

        classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        class_correct = [0] * 10
        class_total = [0] * 10

        for i in range(0, num_samples, batch_size):
            X_batch = X_test[i : i + batch_size]
            y_batch = y_test[i : i + batch_size]

            out = model.forward(X_batch)
            pred = xp.argmax(out, axis=1)

            # Convert to CPU for numpy operations
            pred_cpu = to_cpu(pred)
            y_batch_cpu = to_cpu(y_batch)

            for j in range(len(y_batch_cpu)):
                label = y_batch_cpu[j]
                class_total[label] += 1
                if pred_cpu[j] == label:
                    class_correct[label] += 1

        print("\nPer-class Accuracy:")
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                class_accuracies[classes[i]] = class_acc
                print(f"{classes[i]}: {class_acc:.2f}%")
            else:
                class_accuracies[classes[i]] = 0.0
    else:
        # Simple placeholder for non-verbose mode
        classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        class_accuracies = {cls: 0.0 for cls in classes}

    return {
        "overall_accuracy": accuracy,
        "class_accuracies": class_accuracies,
        "total_samples": total,
        "correct_predictions": correct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test CNN Model Checkpoints on CIFAR-10"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint file")
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "cnn", "resnet32"],
        help="Model type: auto (detect from checkpoint), cnn (SimpleCNN), or resnet32 (ResNet-32)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/cifar-10-batches-py",
        help="CIFAR-10 data directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Test batch size (default: 128)"
    )
    parser.add_argument(
        "--output", type=str, help="Output file to save results (JSON format)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Minimal output (only accuracy percentage)"
    )

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1

    if verbose:
        print(f"Testing checkpoint: {args.checkpoint}")
        print("-" * 50)

    try:
        # Load test data
        if verbose:
            print(f"Loading CIFAR-10 test data from {args.data}...")

        _, _, X_test, y_test = load_cifar10(args.data)

        if verbose:
            print(f"Test set: {X_test.shape}")

        # Determine model type
        model_type = args.model
        if model_type == "auto":
            # Try to detect from checkpoint filename or path
            checkpoint_name = os.path.basename(args.checkpoint).lower()
            if "resnet" in checkpoint_name or "resnet32" in checkpoint_name:
                model_type = "resnet32"
            else:
                model_type = "cnn"  # Default to SimpleCNN

            if verbose:
                print(f"Auto-detected model type: {model_type}")

        # Create model instance
        if model_type == "cnn":
            model = SimpleCNN(num_classes=10)
            if verbose:
                print("Created SimpleCNN model")
        elif model_type == "resnet32":
            model = ResNet32(num_classes=10)
            if verbose:
                print("Created ResNet-32 model")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load checkpoint
        if verbose:
            print("Loading checkpoint...")

        checkpoint_acc = load_checkpoint(model, args.checkpoint)

        if verbose:
            print("Model loaded successfully")
            if checkpoint_acc is not None:
                print(f"Checkpoint accuracy: {checkpoint_acc:.2f}%")

        # Calculate model parameters
        total_params = 0
        if verbose:
            print("Calculating model parameters...")

        # Use a more efficient parameter counting method
        def count_parameters(obj, prefix=""):
            count = 0
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr_value = getattr(obj, attr_name)
                    if hasattr(attr_value, "size") and hasattr(attr_value, "shape"):
                        count += attr_value.size
                        if verbose and len(prefix) == 0:  # Only print for top level
                            print(
                                f"  {attr_name}: {attr_value.shape} -> {attr_value.size:,} params"
                            )
                except (AttributeError, TypeError, ValueError):
                    # Skip attributes that cause issues
                    continue
            return count

        if hasattr(model, "blocks"):
            # ResNet model - count main parameters
            total_params += count_parameters(model)

            # Count block parameters
            if verbose:
                print(f"Counting parameters in {len(model.blocks)} blocks...")
            for i, block in enumerate(model.blocks):
                block_params = count_parameters(block, f"block_{i}")
                total_params += block_params
        else:
            # SimpleCNN model
            total_params = count_parameters(model)

        if verbose:
            print(f"Model: {model.name}, Parameters: {total_params:,}")
            print("-" * 50)

        # Evaluate model
        results = evaluate_model(model, X_test, y_test, args.batch_size, verbose)

        # Add metadata
        results["checkpoint_path"] = args.checkpoint
        results["model_type"] = model_type
        results["model_name"] = model.name
        results["total_parameters"] = total_params
        results["test_time"] = datetime.now().isoformat()
        results["checkpoint_accuracy"] = checkpoint_acc

        # Save results if output file specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            if verbose:
                print(f"\nResults saved to: {args.output}")

        # Print summary
        if args.quiet:
            print(f"{results['overall_accuracy']:.2f}")
        elif not verbose:
            print(f"Test Accuracy: {results['overall_accuracy']:.2f}%")

        if verbose:
            print("\nSummary:")
            print(f"Model: {results['model_name']} ({total_params:,} parameters)")
            print(f"Test Accuracy: {results['overall_accuracy']:.2f}%")
            if checkpoint_acc is not None:
                print(f"Checkpoint Accuracy: {checkpoint_acc:.2f}%")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
