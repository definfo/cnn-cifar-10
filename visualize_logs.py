import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from pathlib import Path


class LogVisualizer:
    def __init__(self):
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.per_class_data = None

    def parse_log_file(self, filepath):
        """Parse log file and extract training metrics"""
        data = []
        with open(filepath, "r") as f:
            for line in f:
                # Training log format: [Epoch X] Average Loss: Y.YYYY
                epoch_match = re.search(r"\[Epoch (\d+)\] Average Loss: ([\d.]+)", line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    loss = float(epoch_match.group(2))
                    data.append({"epoch": epoch, "loss": loss, "accuracy": None})
                    continue

                # Test accuracy format: [Test] Accuracy: XXXX/10000 (XX.XX%)
                test_acc_match = re.search(
                    r"\[Test\] Accuracy: \d+/\d+ \(([\d.]+)%\)", line
                )
                if test_acc_match and data:
                    # Add accuracy to the last epoch entry
                    data[-1]["accuracy"] = float(test_acc_match.group(1))
                    continue

                # Alternative formats for backward compatibility
                epoch_alt = re.search(r"Epoch (\d+)", line)
                loss_alt = re.search(r"Loss: ([\d.]+)", line)
                acc_alt = re.search(r"Acc(?:uracy)?: ([\d.]+)", line)

                if epoch_alt and loss_alt:
                    epoch = int(epoch_alt.group(1))
                    loss = float(loss_alt.group(1))
                    accuracy = float(acc_alt.group(1)) if acc_alt else None
                    data.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})

        return pd.DataFrame(data)

    def parse_test_log_file(self, filepath):
        """Parse test-only log file with different format"""
        data = []
        with open(filepath, "r") as f:
            content = f.read()

            # Extract overall test accuracy: "Overall Test Accuracy: XX.XX%"
            overall_acc_match = re.search(r"Overall Test Accuracy: ([\d.]+)%", content)
            if overall_acc_match:
                accuracy = float(overall_acc_match.group(1))
                data.append(
                    {
                        "epoch": "final",  # Mark as final test result
                        "loss": None,
                        "accuracy": accuracy,
                        "type": "test_only",
                    }
                )

            # Extract per-class accuracies for detailed analysis
            per_class_matches = re.findall(r"(\w+): ([\d.]+)%", content)
            class_data = {}
            for class_name, acc in per_class_matches:
                if class_name in [
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
                ]:
                    class_data[class_name] = float(acc)

            if class_data:
                data.append(
                    {
                        "epoch": "per_class",
                        "loss": None,
                        "accuracy": None,
                        "type": "per_class",
                        "class_accuracies": class_data,
                    }
                )

        return pd.DataFrame(data)

    def load_logs(self, train_log_path="log/train.log", test_log_path="log/test.log"):
        """Load training and test logs"""
        train_log_path, test_log_path = Path(train_log_path), Path(test_log_path)

        if train_log_path.exists():
            self.train_data = self.parse_log_file(train_log_path)
            print(f"Loaded {len(self.train_data)} training records")

            # Debug: show sample of parsed data
            if not self.train_data.empty:
                print("Sample training data:")
                print(self.train_data.head(3).to_string())
        else:
            print(f"Training log file not found: {train_log_path}")
            exit(1)

        if test_log_path.exists():
            # Check if it's a test-only log or training log with test data
            with open(test_log_path, "r") as f:
                content = f.read()

            if "Overall Test Accuracy:" in content:
                # This is a test-only log file
                test_only_data = self.parse_test_log_file(test_log_path)
                print(f"Loaded test-only results: {len(test_only_data)} records")

                # Store per-class data for later visualization
                if not test_only_data.empty:
                    for _, row in test_only_data.iterrows():
                        if row.get("type") == "per_class" and "class_accuracies" in row:
                            self.per_class_data = row["class_accuracies"]
                        elif row.get("type") == "test_only":
                            print(f"Final test accuracy: {row['accuracy']:.2f}%")

                self.test_data = pd.DataFrame()  # No epoch-wise test data
            else:
                # Regular training log format
                self.test_data = self.parse_log_file(test_log_path)
                print(f"Loaded {len(self.test_data)} test records")
        else:
            print(f"Test log file not found: {test_log_path}")
            print("Continuing with training data only...")
            self.test_data = pd.DataFrame()

    def create_per_class_plot(
        self, class_accuracies, save_path="per_class_accuracy.png"
    ):
        """Create a bar plot for per-class accuracies"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        classes = list(class_accuracies.keys())
        accuracies = list(class_accuracies.values())

        bars = ax.bar(classes, accuracies, color="skyblue", edgecolor="navy", alpha=0.7)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title(
            "Per-Class Test Accuracy (CIFAR-10)", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, max(accuracies) * 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Per-class accuracy plot saved to {save_path}")
        plt.close()

    def plot_training_curves(self, save_path="training_curves.png"):
        """Plot training and validation curves"""
        if self.train_data.empty:
            print("No training data available for plotting")
            return

        # Adjust subplot layout based on available data
        has_test_data = not self.test_data.empty
        has_accuracy = (
            "accuracy" in self.train_data.columns
            and self.train_data["accuracy"].notna().any()
        )

        if has_test_data and has_accuracy:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        elif has_accuracy:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes = axes.reshape(1, 2) if len(axes.shape) == 1 else axes

        fig.suptitle("CNN CIFAR-10 Training Progress", fontsize=16)

        # Loss curves
        plot_idx = (0, 0) if axes.ndim == 2 else 0
        ax_loss = axes[plot_idx] if axes.ndim == 2 else axes[0]

        ax_loss.plot(
            self.train_data["epoch"],
            self.train_data["loss"],
            label="Training Loss",
            color="blue",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        if has_test_data:
            ax_loss.plot(
                self.test_data["epoch"],
                self.test_data["loss"],
                label="Test Loss",
                color="red",
                linewidth=2,
                marker="s",
                markersize=4,
            )
        ax_loss.set_title("Loss Over Time")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Accuracy curves (only if accuracy data exists)
        if has_accuracy:
            plot_idx = (0, 1) if axes.ndim == 2 else 1
            ax_acc = axes[plot_idx] if axes.ndim == 2 else axes[1]

            train_acc = self.train_data["accuracy"].dropna()
            train_epochs = self.train_data.loc[train_acc.index, "epoch"]

            ax_acc.plot(
                train_epochs,
                train_acc,
                label="Training Accuracy",
                color="blue",
                linewidth=2,
                marker="o",
                markersize=4,
            )
            if has_test_data and "accuracy" in self.test_data.columns:
                test_acc = self.test_data["accuracy"].dropna()
                test_epochs = self.test_data.loc[test_acc.index, "epoch"]
                ax_acc.plot(
                    test_epochs,
                    test_acc,
                    label="Test Accuracy",
                    color="red",
                    linewidth=2,
                    marker="s",
                    markersize=4,
                )
            ax_acc.set_title("Accuracy Over Time")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy (%)")
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)
        else:
            # Loss smoothing plot if no accuracy
            plot_idx = (0, 1) if axes.ndim == 2 else 1
            ax_smooth = axes[plot_idx] if axes.ndim == 2 else axes[1]

            epochs = self.train_data["epoch"].values
            loss_smoothed = (
                pd.Series(self.train_data["loss"])
                .rolling(window=min(5, len(self.train_data)))
                .mean()
            )
            ax_smooth.plot(
                epochs,
                loss_smoothed,
                color="purple",
                linewidth=2,
                marker="d",
                markersize=4,
            )
            ax_smooth.set_title("Smoothed Training Loss")
            ax_smooth.set_xlabel("Epoch")
            ax_smooth.set_ylabel("Smoothed Loss")
            ax_smooth.grid(True, alpha=0.3)

        # Additional plots only if we have 2x2 layout
        if axes.ndim == 2 and axes.shape == (2, 2):
            # Loss distribution
            axes[1, 0].hist(
                self.train_data["loss"],
                bins=min(20, len(self.train_data) // 2),
                alpha=0.7,
                label="Training Loss",
                color="blue",
                edgecolor="black",
            )
            if has_test_data:
                axes[1, 0].hist(
                    self.test_data["loss"],
                    bins=min(20, len(self.test_data) // 2),
                    alpha=0.7,
                    label="Test Loss",
                    color="red",
                    edgecolor="black",
                )
            axes[1, 0].set_title("Loss Distribution")
            axes[1, 0].set_xlabel("Loss Value")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].legend()

            # Learning progress
            epochs = self.train_data["epoch"].values
            loss_smoothed = (
                pd.Series(self.train_data["loss"])
                .rolling(window=min(3, len(self.train_data)))
                .mean()
            )
            axes[1, 1].plot(
                epochs,
                loss_smoothed,
                color="purple",
                linewidth=2,
                marker="d",
                markersize=4,
            )
            axes[1, 1].set_title(
                f"Smoothed Training Loss ({min(3, len(self.train_data))}-epoch avg)"
            )
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Smoothed Loss")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
        # Remove plt.show() to avoid warning
        plt.close()

    def plot_comparison_metrics(self, save_path="comparison_metrics.png"):
        """Plot side-by-side comparison of train vs test metrics"""
        if self.train_data.empty:
            print("No training data available for comparison")
            return

        if self.test_data.empty:
            print("No test data available - creating training-only analysis")
            self.plot_training_only_analysis(save_path)
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Final epoch comparison
        train_final = self.train_data.iloc[-1] if not self.train_data.empty else None
        test_final = self.test_data.iloc[-1] if not self.test_data.empty else None

        if train_final is not None and test_final is not None:
            metrics = ["loss", "accuracy"]
            train_vals = [train_final["loss"], train_final.get("accuracy", 0)]
            test_vals = [test_final["loss"], test_final.get("accuracy", 0)]

            x = np.arange(len(metrics))
            width = 0.35

            axes[0].bar(x - width / 2, train_vals, width, label="Training", alpha=0.8)
            axes[0].bar(x + width / 2, test_vals, width, label="Test", alpha=0.8)
            axes[0].set_xlabel("Metrics")
            axes[0].set_ylabel("Values")
            axes[0].set_title("Final Epoch Comparison")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(metrics)
            axes[0].legend()

        # Overfitting analysis
        if (
            "accuracy" in self.train_data.columns
            and "accuracy" in self.test_data.columns
        ):
            # Merge data by epoch
            merged = pd.merge(
                self.train_data,
                self.test_data,
                on="epoch",
                suffixes=("_train", "_test"),
            )
            gap = merged["accuracy_train"] - merged["accuracy_test"]

            axes[1].plot(merged["epoch"], gap, color="orange", linewidth=2)
            axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
            axes[1].set_title("Overfitting Analysis\n(Train Accuracy - Test Accuracy)")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy Gap")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison metrics saved to {save_path}")
        plt.close()

    def plot_training_only_analysis(self, save_path="training_analysis.png"):
        """Plot analysis when only training data is available"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Training progress over time
        axes[0].plot(
            self.train_data["epoch"],
            self.train_data["loss"],
            color="blue",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axes[0].set_title("Training Loss Progress")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        # Loss improvement rate
        if len(self.train_data) > 1:
            loss_diff = self.train_data["loss"].diff()
            axes[1].plot(
                self.train_data["epoch"].iloc[1:],
                loss_diff.iloc[1:],
                color="green",
                linewidth=2,
                marker="s",
                markersize=4,
            )
            axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.7)
            axes[1].set_title("Loss Improvement Rate\n(Negative = Improvement)")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss Change")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training analysis saved to {save_path}")
        plt.close()

    def generate_summary_report(self, save_path="training_summary.txt"):
        """Generate a text summary of training results"""
        with open(save_path, "w") as f:
            f.write("CNN CIFAR-10 Training Summary Report\n")
            f.write("=" * 40 + "\n\n")

            if not self.train_data.empty:
                f.write("Training Statistics:\n")
                f.write(f"Total epochs: {len(self.train_data)}\n")
                f.write(
                    f"Final training loss: {self.train_data['loss'].iloc[-1]:.4f}\n"
                )
                f.write(f"Best training loss: {self.train_data['loss'].min():.4f}\n")

                if (
                    "accuracy" in self.train_data.columns
                    and self.train_data["accuracy"].notna().any()
                ):
                    f.write(
                        f"Final training accuracy: {self.train_data['accuracy'].iloc[-1]:.2f}%\n"
                    )
                    f.write(
                        f"Best training accuracy: {self.train_data['accuracy'].max():.2f}%\n"
                    )
                f.write("\n")
            else:
                f.write("No training data available\n\n")

            if not self.test_data.empty:
                f.write("Test Statistics:\n")
                f.write(f"Final test loss: {self.test_data['loss'].iloc[-1]:.4f}\n")
                f.write(f"Best test loss: {self.test_data['loss'].min():.4f}\n")

                if (
                    "accuracy" in self.test_data.columns
                    and self.test_data["accuracy"].notna().any()
                ):
                    f.write(
                        f"Final test accuracy: {self.test_data['accuracy'].iloc[-1]:.2f}%\n"
                    )
                    f.write(
                        f"Best test accuracy: {self.test_data['accuracy'].max():.2f}%\n"
                    )
            else:
                f.write("No test data available\n")

        print(f"Summary report saved to {save_path}")

    def has_per_class_data(self):
        """Check if per-class accuracy data is available"""
        return self.per_class_data is not None


def main():
    parser = argparse.ArgumentParser(
        description="Visualize train/test logs for CNN on CIFAR-10"
    )
    parser.add_argument(
        "--train-log",
        type=str,
        default="log/train.log",
        help="Path to train log file (default: log/train.log)",
    )
    parser.add_argument(
        "--test-log",
        type=str,
        default="log/test.log",
        help="Path to test log file (default: log/test.log)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="model",
        help="Prefix for output files (default: model)",
    )
    args = parser.parse_args()

    # Initialize visualizer
    visualizer = LogVisualizer()

    # Load log files (relative paths under `log`)
    visualizer.load_logs(args.train_log, args.test_log)

    # Generate visualizations
    visualizer.plot_training_curves(f"{args.prefix}_training_progress.png")
    visualizer.plot_comparison_metrics(f"{args.prefix}_comparison.png")
    visualizer.generate_summary_report(f"{args.prefix}_training_summary.txt")

    # Generate per-class plot if data is available
    if visualizer.has_per_class_data():
        visualizer.create_per_class_plot(
            visualizer.per_class_data, f"{args.prefix}_per_class_accuracy.png"
        )

    print("Visualizations completed!")
    print("Generated files:")
    print(f"- {args.prefix}_training_progress.png")
    print(f"- {args.prefix}_comparison.png (or {args.prefix}_training_analysis.png)")
    print(f"- {args.prefix}_training_summary.txt")
    if visualizer.has_per_class_data():
        print(f"- {args.prefix}_per_class_accuracy.png")


if __name__ == "__main__":
    main()
