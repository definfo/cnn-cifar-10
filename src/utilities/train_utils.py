import os
import pickle
import numpy as np
from utilities.backend import xp, to_cpu, convert_array_to_current_backend

from tqdm import tqdm as tqdm_cls


class CrossBackendUnpickler(pickle.Unpickler):
    """Custom unpickler to handle CuPy/NumPy cross-compatibility"""

    def find_class(self, module, name):
        # Handle all CuPy-related modules by redirecting to NumPy
        if module.startswith("cupy"):
            if name == "ndarray":
                return np.ndarray
            elif name in ["dtype", "int64", "float32", "float64", "int32"]:
                return getattr(np, name)
            else:
                # For other CuPy classes, try to find NumPy equivalent
                try:
                    return getattr(np, name)
                except AttributeError:
                    # If no NumPy equivalent, return a dummy class
                    return type(f"Dummy{name}", (), {})
        return super().find_class(module, name)

    def load(self):
        """Override load to handle CuPy arrays that can't be unpickled"""
        try:
            return super().load()
        except Exception as e:
            # If unpickling fails completely, try alternative approach
            raise RuntimeError(f"Failed to unpickle checkpoint: {e}")


def safe_load_checkpoint(path):
    """
    Safely load checkpoint with multiple fallback strategies
    """
    with open(path, "rb") as f:
        # Strategy 1: Try normal pickle loading
        try:
            f.seek(0)
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if "cupy" not in str(e).lower():
                raise e

        # Strategy 2: Try custom unpickler
        try:
            f.seek(0)
            unpickler = CrossBackendUnpickler(f)
            return unpickler.load()
        except Exception:
            pass

        # Strategy 3: Use pickle with protocol 0 (more compatible)
        try:
            f.seek(0)
            # Read raw data and try to reconstruct
            _ = f.read()
            f.seek(0)

            # Try to load with different protocols
            for protocol in [0, 1, 2]:
                try:
                    f.seek(0)
                    return pickle.load(f)
                except Exception:
                    continue

        except Exception:
            pass

        # If all else fails, raise informative error
        raise RuntimeError(
            f"Cannot load checkpoint {path}. "
            "The checkpoint was likely saved with CuPy but CuPy is not available. "
            "Please install CuPy or recreate the checkpoint with NumPy backend."
        )


def load_cifar10_batch(filename):
    # Load a single batch from CIFAR-10 python format
    with open(filename, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        X = batch[b"data"]
        y = batch[b"labels"]
        X = X.reshape(-1, 3, 32, 32).astype("float32") / 255.0
        y = xp.array(y, dtype=xp.int64)
        X = xp.array(X, dtype=xp.float32)
    return X, y


def load_cifar10(data_dir):
    # Load all training and test data
    X_train, y_train = [], []
    for i in range(1, 6):
        X, y = load_cifar10_batch(os.path.join(data_dir, f"data_batch_{i}"))
        X_train.append(X)
        y_train.append(y)
    X_train = xp.concatenate(X_train)
    y_train = xp.concatenate(y_train)
    X_test, y_test = load_cifar10_batch(os.path.join(data_dir, "test_batch"))
    return X_train, y_train, X_test, y_test


def save_checkpoint(model, path, accuracy=None):
    # Save model parameters and accuracy using pickle (move to CPU first)
    state = {}

    # Handle ResNet models with blocks
    if hasattr(model, "blocks"):
        # Save main model parameters
        for attr_name in dir(model):
            if not attr_name.startswith("_") and attr_name != "blocks":
                attr_value = getattr(model, attr_name)
                if hasattr(attr_value, "shape") or isinstance(
                    attr_value, (int, float, str, bool)
                ):
                    state[attr_name] = (
                        to_cpu(attr_value)
                        if hasattr(attr_value, "shape")
                        else attr_value
                    )

        # Save block parameters separately
        state["blocks_data"] = []
        for i, block in enumerate(model.blocks):
            block_state = {}
            for attr_name in dir(block):
                if not attr_name.startswith("_"):
                    attr_value = getattr(block, attr_name)
                    if hasattr(attr_value, "shape"):
                        block_state[attr_name] = to_cpu(attr_value)
                    elif isinstance(attr_value, (int, float, str, bool)):
                        block_state[attr_name] = attr_value
            state["blocks_data"].append(block_state)
    else:
        # Handle simple models (SimpleCNN)
        state = {k: to_cpu(v) for k, v in model.__dict__.items()}

    if accuracy is not None:
        state["_checkpoint_accuracy"] = accuracy
    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"[Checkpoint] Model saved to {path}")


def load_checkpoint(model, path):
    # Load model parameters using pickle and return accuracy if available
    print("[WARNING] Attempting to load checkpoint that may contain CuPy arrays...")

    params = safe_load_checkpoint(path)
    accuracy = params.pop("_checkpoint_accuracy", None)

    # Handle ResNet models with blocks
    if hasattr(model, "blocks") and "blocks_data" in params:
        blocks_data = params.pop("blocks_data")

        # Load main model parameters
        for k, v in params.items():
            if hasattr(model, k):
                current_attr = getattr(model, k)
                if hasattr(current_attr, "shape"):
                    # Ensure we convert any CuPy arrays to current backend
                    converted_v = convert_array_to_current_backend(v)
                    setattr(model, k, converted_v)
                else:
                    setattr(model, k, v)

        # Load block parameters
        for i, block_state in enumerate(blocks_data):
            if i < len(model.blocks):
                block = model.blocks[i]
                for k, v in block_state.items():
                    if hasattr(block, k):
                        current_attr = getattr(block, k)
                        if hasattr(current_attr, "shape"):
                            converted_v = convert_array_to_current_backend(v)
                            setattr(block, k, converted_v)
                        else:
                            setattr(block, k, v)
    else:
        # Handle simple models (SimpleCNN)
        for k, v in params.items():
            if hasattr(model, k):
                current_attr = getattr(model, k)
                if hasattr(current_attr, "shape"):
                    converted_v = convert_array_to_current_backend(v)
                    setattr(model, k, converted_v)
                else:
                    setattr(model, k, v)

    print(f"[Checkpoint] Model loaded from {path}")
    return accuracy


def cross_entropy_loss(pred, target):
    # pred: (N, num_classes), target: (N,)
    N = pred.shape[0]
    # Clip for numerical stability
    pred = xp.clip(pred, 1e-9, 1 - 1e-9)
    log_likelihood = -xp.log(pred[xp.arange(N), target])
    loss = xp.sum(log_likelihood) / N
    return loss


def cross_entropy_grad(pred, target):
    # Gradient of softmax + cross-entropy
    N = pred.shape[0]
    grad = pred.copy()
    grad[xp.arange(N), target] -= 1
    grad /= N
    return grad


def sgd_update(param, grad, lr):
    # Simple SGD update
    param -= lr * grad
    return param


class LearningRateScheduler:
    """Learning rate scheduler with warmup, linear decay, and cosine annealing."""

    def __init__(
        self,
        base_lr,
        total_epochs,
        warmup_epochs=0,
        decay_type="cosine",
        min_lr=1e-6,
        decay_factor=0.1,
        decay_steps=None,
    ):
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.decay_type = decay_type.lower()
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps or []

    def get_lr(self, epoch):
        """Get learning rate for given epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr * (epoch + 1) / self.warmup_epochs

        # Adjust epoch for post-warmup scheduling
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs

        if self.decay_type == "cosine":
            # Cosine annealing
            lr = (
                self.min_lr
                + (self.base_lr - self.min_lr)
                * (1 + xp.cos(xp.pi * adjusted_epoch / adjusted_total))
                / 2
            )
        elif self.decay_type == "linear":
            # Linear decay
            lr = self.base_lr * (1 - adjusted_epoch / adjusted_total)
            lr = max(lr, self.min_lr)
        elif self.decay_type == "step":
            # Step decay
            lr = self.base_lr
            for step in self.decay_steps:
                if epoch >= step:
                    lr *= self.decay_factor
            lr = max(lr, self.min_lr)
        else:
            # Constant learning rate
            lr = self.base_lr

        return float(lr)


class DynamicDropout:
    """Dynamic dropout that adapts during training."""

    def __init__(
        self,
        initial_rate=0.5,
        min_rate=0.1,
        max_rate=0.7,
        strategy="linear_decay",
        plateau_patience=5,
    ):
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.strategy = strategy.lower()
        self.plateau_patience = plateau_patience
        self.best_acc = 0.0
        self.plateau_count = 0

    def get_dropout_rate(self, epoch, total_epochs, current_acc=None):
        """Get dropout rate for given epoch and performance."""
        if self.strategy == "linear_decay":
            # Linear decay from initial to min
            progress = epoch / total_epochs
            rate = self.initial_rate * (1 - progress) + self.min_rate * progress
        elif self.strategy == "cosine_decay":
            # Cosine decay
            progress = epoch / total_epochs
            rate = (
                self.min_rate
                + (self.initial_rate - self.min_rate)
                * (1 + xp.cos(xp.pi * progress))
                / 2
            )
        elif self.strategy == "adaptive":
            # Adaptive based on performance
            if current_acc is not None:
                if current_acc > self.best_acc:
                    self.best_acc = current_acc
                    self.plateau_count = 0
                    # Reduce dropout when improving
                    rate = max(self.initial_rate * 0.9, self.min_rate)
                else:
                    self.plateau_count += 1
                    if self.plateau_count >= self.plateau_patience:
                        # Increase dropout when plateauing
                        rate = min(self.initial_rate * 1.2, self.max_rate)
                    else:
                        rate = self.initial_rate
            else:
                rate = self.initial_rate
        else:
            # Constant dropout
            rate = self.initial_rate

        return max(self.min_rate, min(rate, self.max_rate))


def train(
    model,
    X_train,
    y_train,
    batch_size,
    lr,
    _epoch,
    checkpoint_dir=None,
    model_name=None,
    X_test=None,
    y_test=None,
    test_batch_size=32,
    lr_scheduler=None,
    dropout_scheduler=None,
) -> float:
    num_samples = X_train.shape[0]
    indices = xp.arange(num_samples)
    indices = xp.random.permutation(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    running_loss = 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size
    batch_iter = range(num_batches)
    pbar = tqdm_cls(batch_iter, desc=f"Epoch {_epoch + 1}", ncols=80, leave=False)

    # Set model to training mode
    model.train()

    # Update learning rate if scheduler is provided
    current_lr = lr
    if lr_scheduler is not None:
        current_lr = lr_scheduler.get_lr(_epoch)
        print(f"[Epoch {_epoch + 1}] Learning rate: {current_lr:.6f}")

    for batch_idx in batch_iter:
        i = batch_idx * batch_size
        X_batch = X_train[i : i + batch_size]
        y_batch = y_train[i : i + batch_size]
        out = model.forward(X_batch)

        # Use cross_entropy_loss from model.cnn module
        from model.cnn import cross_entropy_loss

        loss_val = cross_entropy_loss(out, y_batch)
        running_loss += float(to_cpu(loss_val))
        model.zero_grad()
        model.backward(y_batch)
        model.step(current_lr)
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=running_loss / (batch_idx + 1), lr=current_lr)
    if pbar is not None:
        pbar.close()
    if pbar is None:
        print(f"[Epoch {_epoch + 1}] Loss: {running_loss / num_batches:.4f}")
    avg_loss = running_loss / num_batches
    print(f"[Epoch {_epoch + 1}] Average Loss: {avg_loss:.4f}")

    accuracy = test(model, X_test, y_test, test_batch_size or batch_size)

    # Update dropout rate if scheduler is provided
    if dropout_scheduler is not None:
        old_dropout = model.dropout_rate
        # Pass total epochs - need to get from somewhere, for now use a reasonable default
        total_epochs = getattr(model, "_total_epochs", 100)
        new_dropout = dropout_scheduler.get_dropout_rate(_epoch, total_epochs, accuracy)
        if abs(new_dropout - old_dropout) > 0.01:  # Only update if significant change
            model.dropout_rate = new_dropout
            print(
                f"[Epoch {_epoch + 1}] Dropout rate updated: {old_dropout:.3f} -> {new_dropout:.3f}"
            )

    # Save checkpoint for this epoch
    if checkpoint_dir is not None:
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save epoch checkpoint
        model_name = model_name if model_name is not None else model.name
        epoch_checkpoint_path = os.path.join(
            checkpoint_dir, f"{model_name}_epoch{_epoch + 1}.pkl"
        )
        save_checkpoint(model, epoch_checkpoint_path, accuracy=accuracy)

    return accuracy


def test(model, X_test, y_test, batch_size):
    # Set model to evaluation mode
    model.eval()

    num_samples = X_test.shape[0]
    correct = 0
    total = 0
    for i in range(0, num_samples, batch_size):
        X_batch = X_test[i : i + batch_size]
        y_batch = y_test[i : i + batch_size]
        out = model.forward(X_batch)
        pred = xp.argmax(out, axis=1)
        correct += int(xp.sum(pred == y_batch))
        total += y_batch.shape[0]
    accuracy = 100.0 * correct / total
    print(f"[Test] Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    return accuracy
