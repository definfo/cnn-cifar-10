from utilities.backend import xp

# Helper functions for layers


def relu(x):
    return xp.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(xp.float32)


def softmax(x):
    e_x = xp.exp(x - xp.max(x, axis=1, keepdims=True))
    return e_x / xp.sum(e_x, axis=1, keepdims=True)


def cross_entropy_loss(pred, target):
    N = pred.shape[0]
    pred = xp.clip(pred, 1e-9, 1 - 1e-9)
    log_likelihood = -xp.log(pred[xp.arange(N), target])
    loss = xp.sum(log_likelihood) / N
    return loss


def cross_entropy_grad(pred, target):
    N = pred.shape[0]
    grad = pred.copy()
    grad[xp.arange(N), target] -= 1
    grad /= N
    return grad


def im2col(x, kernel_size, stride=1, padding=0):
    """Convert input to column matrix for efficient convolution."""
    N, C, H, W = x.shape
    k = kernel_size
    out_h = (H + 2 * padding - k) // stride + 1
    out_w = (W + 2 * padding - k) // stride + 1

    if padding > 0:
        x = xp.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )

    col = xp.zeros((N, C, k, k, out_h, out_w), dtype=x.dtype)
    for j in range(k):
        j_max = j + stride * out_h
        for i in range(k):
            i_max = i + stride * out_w
            col[:, :, j, i, :, :] = x[:, :, j:j_max:stride, i:i_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, x_shape, kernel_size, stride=1, padding=0):
    """Convert column matrix back to input format."""
    N, C, H, W = x_shape
    k = kernel_size
    out_h = (H + 2 * padding - k) // stride + 1
    out_w = (W + 2 * padding - k) // stride + 1

    col = col.reshape(N, out_h, out_w, C, k, k).transpose(0, 3, 4, 5, 1, 2)
    x = xp.zeros(
        (N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1),
        dtype=col.dtype,
    )

    for j in range(k):
        j_max = j + stride * out_h
        for i in range(k):
            i_max = i + stride * out_w
            x[:, :, j:j_max:stride, i:i_max:stride] += col[:, :, j, i, :, :]

    return x[:, :, padding : H + padding, padding : W + padding]


def batch_norm_forward(x, gamma, beta, eps=1e-8):
    """Batch normalization forward pass."""
    mean = xp.mean(x, axis=0, keepdims=True)
    var = xp.var(x, axis=0, keepdims=True)
    x_norm = (x - mean) / xp.sqrt(var + eps)
    out = gamma * x_norm + beta
    cache = (x, x_norm, mean, var, gamma, eps)
    return out, cache


def batch_norm_backward(dout, cache):
    """Batch normalization backward pass."""
    x, x_norm, mean, var, gamma, eps = cache
    N = x.shape[0]

    dgamma = xp.sum(dout * x_norm, axis=0, keepdims=True)
    dbeta = xp.sum(dout, axis=0, keepdims=True)

    dx_norm = dout * gamma
    dvar = xp.sum(
        dx_norm * (x - mean) * -0.5 * (var + eps) ** -1.5, axis=0, keepdims=True
    )
    dmean = xp.sum(
        dx_norm * -1 / xp.sqrt(var + eps), axis=0, keepdims=True
    ) + dvar * xp.mean(-2 * (x - mean), axis=0, keepdims=True)
    dx = dx_norm / xp.sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_rate=0.5, training=True):
    """Dropout forward pass."""
    if not training:
        return x, None

    mask = xp.random.rand(*x.shape) > dropout_rate
    out = x * mask / (1 - dropout_rate)  # Scale to maintain expected value
    return out, mask


def dropout_backward(dout, mask, dropout_rate=0.5):
    """Dropout backward pass."""
    if mask is None:
        return dout
    return dout * mask / (1 - dropout_rate)


class AdamOptimizer:
    """ADAM optimizer implementation."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # time step
        self.m = {}  # first moment
        self.v = {}  # second moment

    def update(self, param_name, param, grad):
        """Update parameter using ADAM."""
        if param_name not in self.m:
            self.m[param_name] = xp.zeros_like(param)
            self.v[param_name] = xp.zeros_like(param)

        # Update biased first moment estimate
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad

        # Update biased second raw moment estimate
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (
            grad**2
        )

        # Compute bias-corrected first moment estimate
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)

        # Update parameters
        param -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

        return param

    def step(self):
        """Increment time step."""
        self.t += 1


class SimpleCNN:
    """
    A minimal CNN for CIFAR-10 using xp (NumPy or CuPy). Supports forward and backward.
    Architecture: Conv -> BatchNorm -> ReLU -> MaxPool -> FC -> BatchNorm -> ReLU -> Dropout -> FC -> Softmax
    """

    def __init__(
        self,
        num_classes=10,
        dropout_rate=0.5,
        optimizer="sgd",
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
    ):
        self.name = "SimpleCNN"
        self.dropout_rate = dropout_rate
        self.training = True
        self.optimizer_type = optimizer

        # Conv: 3x32x32 -> 8x30x30 (kernel=3, stride=1, no padding)
        self.conv_w = xp.random.randn(8, 3, 3, 3).astype(xp.float32) * 0.1
        self.conv_b = xp.zeros(8, dtype=xp.float32)

        # Batch norm for conv layer
        self.conv_bn_gamma = xp.ones((1, 8, 1, 1), dtype=xp.float32)
        self.conv_bn_beta = xp.zeros((1, 8, 1, 1), dtype=xp.float32)

        # FC1: flatten 8x15x15 (after 2x2 maxpool) -> 128
        self.fc1_w = xp.random.randn(8 * 15 * 15, 128).astype(xp.float32) * 0.1
        self.fc1_b = xp.zeros(128, dtype=xp.float32)

        # Batch norm for fc1 layer
        self.fc1_bn_gamma = xp.ones(128, dtype=xp.float32)
        self.fc1_bn_beta = xp.zeros(128, dtype=xp.float32)

        # FC2: 128 -> num_classes
        self.fc2_w = xp.random.randn(128, num_classes).astype(xp.float32) * 0.1
        self.fc2_b = xp.zeros(num_classes, dtype=xp.float32)

        # Gradients
        self.zero_grad()

        # Initialize optimizer
        if optimizer.lower() == "adam":
            self.optimizer = AdamOptimizer(lr=lr, beta1=beta1, beta2=beta2)
        else:
            self.optimizer = None  # Use SGD
            self.lr = lr

    def zero_grad(self):
        self.d_conv_w = xp.zeros_like(self.conv_w)
        self.d_conv_b = xp.zeros_like(self.conv_b)
        self.d_conv_bn_gamma = xp.zeros_like(self.conv_bn_gamma)
        self.d_conv_bn_beta = xp.zeros_like(self.conv_bn_beta)
        self.d_fc1_w = xp.zeros_like(self.fc1_w)
        self.d_fc1_b = xp.zeros_like(self.fc1_b)
        self.d_fc1_bn_gamma = xp.zeros_like(self.fc1_bn_gamma)
        self.d_fc1_bn_beta = xp.zeros_like(self.fc1_bn_beta)
        self.d_fc2_w = xp.zeros_like(self.fc2_w)
        self.d_fc2_b = xp.zeros_like(self.fc2_b)

    def conv2d(self, x, w, b):
        N, C, H, W = x.shape
        F, _, k, _ = w.shape

        # Vectorized convolution using im2col
        col = im2col(x, k)  # (N*out_h*out_w, C*k*k)
        w_col = w.reshape(F, -1).T  # (C*k*k, F)

        out = col @ w_col + b  # (N*out_h*out_w, F)
        out_h = H - k + 1
        out_w = W - k + 1
        out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)

        return out

    def conv2d_backward(self, x, w, dout):
        N, C, H, W = x.shape
        F, _, k, _ = w.shape
        _, _, out_h, out_w = dout.shape

        # Vectorized backward pass
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, F)  # (N*out_h*out_w, F)

        # Gradient w.r.t weights
        col = im2col(x, k)  # (N*out_h*out_w, C*k*k)
        dw = (col.T @ dout_reshaped).T.reshape(F, C, k, k)  # (F, C, k, k)

        # Gradient w.r.t bias
        db = xp.sum(dout_reshaped, axis=0)  # (F,)

        # Gradient w.r.t input
        w_col = w.reshape(F, -1)  # (F, C*k*k)
        dx_col = dout_reshaped @ w_col  # (N*out_h*out_w, C*k*k)
        dx = col2im(dx_col, x.shape, k)

        return dx, dw, db

    def maxpool2d(self, x, size=2, stride=2):
        N, C, H, W = x.shape
        out_h = (H - size) // stride + 1
        out_w = (W - size) // stride + 1

        # Optimized CPU version using vectorized operations
        x_reshaped = xp.zeros((N, C, out_h, out_w, size, size), dtype=x.dtype)
        for i in range(size):
            for j in range(size):
                x_reshaped[:, :, :, :, i, j] = x[
                    :,
                    :,
                    i : i + out_h * stride : stride,
                    j : j + out_w * stride : stride,
                ]

        x_flat = x_reshaped.reshape(N, C, out_h, out_w, size * size)
        out = x_flat.max(axis=4)
        self.pool_cache = (x, x_flat.argmax(axis=4))
        return out

    def maxpool2d_backward(self, dout, x_shape, size=2, stride=2):
        N, C, H, W = x_shape
        out_h = (H - size) // stride + 1
        out_w = (W - size) // stride + 1

        x, argmax = self.pool_cache
        dx = xp.zeros_like(x)

        # Optimized CPU backward pass
        dx_flat = xp.zeros((N, C, out_h, out_w, size * size), dtype=x.dtype)
        flat_indices = xp.arange(size * size)
        mask = argmax[..., None] == flat_indices
        dx_flat[mask] = dout[..., None].repeat(size * size, axis=4)[mask]

        dx_reshaped = dx_flat.reshape(N, C, out_h, out_w, size, size)
        for i in range(size):
            for j in range(size):
                dx[
                    :,
                    :,
                    i : i + out_h * stride : stride,
                    j : j + out_w * stride : stride,
                ] += dx_reshaped[:, :, :, :, i, j]

        return dx

    def forward(self, x):
        self.x = x
        # Conv -> BatchNorm -> ReLU -> MaxPool
        self.z1 = self.conv2d(x, self.conv_w, self.conv_b)
        self.z1_bn, self.conv_bn_cache = batch_norm_forward(
            self.z1.reshape(-1, self.z1.shape[1]),
            self.conv_bn_gamma.reshape(-1),
            self.conv_bn_beta.reshape(-1),
        )
        self.z1_bn = self.z1_bn.reshape(self.z1.shape)
        self.a1 = relu(self.z1_bn)
        self.p1 = self.maxpool2d(self.a1)

        # FC1 -> BatchNorm -> ReLU -> Dropout
        self.flat = self.p1.reshape(x.shape[0], -1)
        self.z2 = self.flat @ self.fc1_w + self.fc1_b
        self.z2_bn, self.fc1_bn_cache = batch_norm_forward(
            self.z2, self.fc1_bn_gamma, self.fc1_bn_beta
        )
        self.a2 = relu(self.z2_bn)
        self.a2_drop, self.dropout_mask = dropout_forward(
            self.a2, self.dropout_rate, self.training
        )

        # FC2 -> Softmax
        self.z3 = self.a2_drop @ self.fc2_w + self.fc2_b
        self.out = softmax(self.z3)
        return self.out

    def backward(self, y):
        # y: (N,) integer labels
        dout = cross_entropy_grad(self.out, y)  # (N, num_classes)

        # FC2
        self.d_fc2_w = self.a2_drop.T @ dout
        self.d_fc2_b = xp.sum(dout, axis=0)
        da2_drop = dout @ self.fc2_w.T

        # Dropout backward
        da2 = dropout_backward(da2_drop, self.dropout_mask, self.dropout_rate)
        dz2_bn = da2 * relu_deriv(self.z2_bn)

        # FC1 batch norm backward
        dz2, dgamma_fc1, dbeta_fc1 = batch_norm_backward(dz2_bn, self.fc1_bn_cache)
        self.d_fc1_bn_gamma = dgamma_fc1.squeeze()
        self.d_fc1_bn_beta = dbeta_fc1.squeeze()

        # FC1
        self.d_fc1_w = self.flat.T @ dz2
        self.d_fc1_b = xp.sum(dz2, axis=0)
        dflat = dz2 @ self.fc1_w.T
        dp1 = dflat.reshape(self.p1.shape)

        # MaxPool
        da1 = self.maxpool2d_backward(dp1, self.a1.shape)
        dz1_bn = da1 * relu_deriv(self.z1_bn)

        # Conv batch norm backward
        dz1_bn_flat = dz1_bn.reshape(-1, dz1_bn.shape[1])
        dz1_flat, dgamma_conv, dbeta_conv = batch_norm_backward(
            dz1_bn_flat, self.conv_bn_cache
        )
        dz1 = dz1_flat.reshape(self.z1.shape)
        self.d_conv_bn_gamma = dgamma_conv.reshape(self.conv_bn_gamma.shape)
        self.d_conv_bn_beta = dbeta_conv.reshape(self.conv_bn_beta.shape)

        # Conv
        dx, dw, db = self.conv2d_backward(self.x, self.conv_w, dz1)
        self.d_conv_w = dw
        self.d_conv_b = db

    def step(self, lr=None):
        """Update parameters using the specified optimizer."""
        if self.optimizer_type.lower() == "adam":
            self.optimizer.step()
            # Update all parameters using ADAM
            self.conv_w = self.optimizer.update("conv_w", self.conv_w, self.d_conv_w)
            self.conv_b = self.optimizer.update("conv_b", self.conv_b, self.d_conv_b)
            self.conv_bn_gamma = self.optimizer.update(
                "conv_bn_gamma", self.conv_bn_gamma, self.d_conv_bn_gamma
            )
            self.conv_bn_beta = self.optimizer.update(
                "conv_bn_beta", self.conv_bn_beta, self.d_conv_bn_beta
            )
            self.fc1_w = self.optimizer.update("fc1_w", self.fc1_w, self.d_fc1_w)
            self.fc1_b = self.optimizer.update("fc1_b", self.fc1_b, self.d_fc1_b)
            self.fc1_bn_gamma = self.optimizer.update(
                "fc1_bn_gamma", self.fc1_bn_gamma, self.d_fc1_bn_gamma
            )
            self.fc1_bn_beta = self.optimizer.update(
                "fc1_bn_beta", self.fc1_bn_beta, self.d_fc1_bn_beta
            )
            self.fc2_w = self.optimizer.update("fc2_w", self.fc2_w, self.d_fc2_w)
            self.fc2_b = self.optimizer.update("fc2_b", self.fc2_b, self.d_fc2_b)
        else:
            # SGD update
            effective_lr = lr if lr is not None else self.lr
            self.conv_w -= effective_lr * self.d_conv_w
            self.conv_b -= effective_lr * self.d_conv_b
            self.conv_bn_gamma -= effective_lr * self.d_conv_bn_gamma
            self.conv_bn_beta -= effective_lr * self.d_conv_bn_beta
            self.fc1_w -= effective_lr * self.d_fc1_w
            self.fc1_b -= effective_lr * self.d_fc1_b
            self.fc1_bn_gamma -= effective_lr * self.d_fc1_bn_gamma
            self.fc1_bn_beta -= effective_lr * self.d_fc1_bn_beta
            self.fc2_w -= effective_lr * self.d_fc2_w
            self.fc2_b -= effective_lr * self.d_fc2_b

    def train(self):
        """Set model to training mode."""
        self.training = True

    def eval(self):
        """Set model to evaluation mode."""
        self.training = False


class ResidualBlock:
    """Basic residual block for ResNet."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample

        # First conv layer
        self.conv1_w = xp.random.randn(out_channels, in_channels, 3, 3).astype(
            xp.float32
        ) * xp.sqrt(2.0 / (in_channels * 3 * 3))
        self.conv1_b = xp.zeros(out_channels, dtype=xp.float32)
        self.bn1_gamma = xp.ones((1, out_channels, 1, 1), dtype=xp.float32)
        self.bn1_beta = xp.zeros((1, out_channels, 1, 1), dtype=xp.float32)

        # Second conv layer
        self.conv2_w = xp.random.randn(out_channels, out_channels, 3, 3).astype(
            xp.float32
        ) * xp.sqrt(2.0 / (out_channels * 3 * 3))
        self.conv2_b = xp.zeros(out_channels, dtype=xp.float32)
        self.bn2_gamma = xp.ones((1, out_channels, 1, 1), dtype=xp.float32)
        self.bn2_beta = xp.zeros((1, out_channels, 1, 1), dtype=xp.float32)

        # Shortcut connection (if needed)
        if downsample:
            self.shortcut_w = xp.random.randn(out_channels, in_channels, 1, 1).astype(
                xp.float32
            ) * xp.sqrt(2.0 / in_channels)
            self.shortcut_b = xp.zeros(out_channels, dtype=xp.float32)
            self.shortcut_bn_gamma = xp.ones((1, out_channels, 1, 1), dtype=xp.float32)
            self.shortcut_bn_beta = xp.zeros((1, out_channels, 1, 1), dtype=xp.float32)

        # Initialize gradients
        self.zero_grad()

    def zero_grad(self):
        self.d_conv1_w = xp.zeros_like(self.conv1_w)
        self.d_conv1_b = xp.zeros_like(self.conv1_b)
        self.d_bn1_gamma = xp.zeros_like(self.bn1_gamma)
        self.d_bn1_beta = xp.zeros_like(self.bn1_beta)
        self.d_conv2_w = xp.zeros_like(self.conv2_w)
        self.d_conv2_b = xp.zeros_like(self.conv2_b)
        self.d_bn2_gamma = xp.zeros_like(self.bn2_gamma)
        self.d_bn2_beta = xp.zeros_like(self.bn2_beta)

        if self.downsample:
            self.d_shortcut_w = xp.zeros_like(self.shortcut_w)
            self.d_shortcut_b = xp.zeros_like(self.shortcut_b)
            self.d_shortcut_bn_gamma = xp.zeros_like(self.shortcut_bn_gamma)
            self.d_shortcut_bn_beta = xp.zeros_like(self.shortcut_bn_beta)

    def conv2d_with_stride(self, x, w, b, stride=1, padding=1):
        """Convolution with stride and padding support."""
        if padding > 0:
            x = xp.pad(
                x,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode="constant",
            )

        N, C, H, W = x.shape
        F, _, k, _ = w.shape
        out_h = (H - k) // stride + 1
        out_w = (W - k) // stride + 1

        col = im2col(x, k, stride=stride, padding=0)
        w_col = w.reshape(F, -1).T
        out = col @ w_col + b
        out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)
        return out

    def conv2d_backward_with_stride(self, x, w, dout, stride=1, padding=1):
        """Backward pass for convolution with stride and padding."""
        if padding > 0:
            x_padded = xp.pad(
                x,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode="constant",
            )
        else:
            x_padded = x

        N, C, H_pad, W_pad = x_padded.shape
        F, _, k, _ = w.shape
        _, _, out_h, out_w = dout.shape

        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, F)
        col = im2col(x_padded, k, stride=stride, padding=0)

        # Gradients
        dw = (col.T @ dout_reshaped).T.reshape(F, C, k, k)
        db = xp.sum(dout_reshaped, axis=0)

        w_col = w.reshape(F, -1)
        dx_col = dout_reshaped @ w_col
        dx_padded = col2im(dx_col, x_padded.shape, k, stride=stride, padding=0)

        if padding > 0:
            dx = dx_padded[:, :, padding:-padding, padding:-padding]
        else:
            dx = dx_padded

        return dx, dw, db

    def forward(self, x):
        self.x = x

        # First conv + bn + relu
        self.z1 = self.conv2d_with_stride(
            x, self.conv1_w, self.conv1_b, stride=self.stride, padding=1
        )
        self.z1_bn, self.bn1_cache = batch_norm_forward(
            self.z1.reshape(-1, self.z1.shape[1]),
            self.bn1_gamma.reshape(-1),
            self.bn1_beta.reshape(-1),
        )
        self.z1_bn = self.z1_bn.reshape(self.z1.shape)
        self.a1 = relu(self.z1_bn)

        # Second conv + bn
        self.z2 = self.conv2d_with_stride(
            self.a1, self.conv2_w, self.conv2_b, stride=1, padding=1
        )
        self.z2_bn, self.bn2_cache = batch_norm_forward(
            self.z2.reshape(-1, self.z2.shape[1]),
            self.bn2_gamma.reshape(-1),
            self.bn2_beta.reshape(-1),
        )
        self.z2_bn = self.z2_bn.reshape(self.z2.shape)

        # Shortcut connection
        if self.downsample:
            self.shortcut = self.conv2d_with_stride(
                x, self.shortcut_w, self.shortcut_b, stride=self.stride, padding=0
            )
            self.shortcut_bn, self.shortcut_bn_cache = batch_norm_forward(
                self.shortcut.reshape(-1, self.shortcut.shape[1]),
                self.shortcut_bn_gamma.reshape(-1),
                self.shortcut_bn_beta.reshape(-1),
            )
            self.shortcut_bn = self.shortcut_bn.reshape(self.shortcut.shape)
            identity = self.shortcut_bn
        else:
            identity = x

        # Add residual connection and apply ReLU
        self.out = relu(self.z2_bn + identity)
        return self.out

    def backward(self, dout):
        # Backward through final ReLU
        dz2_bn_plus_identity = dout * relu_deriv(
            self.z2_bn + (self.shortcut_bn if self.downsample else self.x)
        )

        # Split gradients for main path and shortcut
        dz2_bn = dz2_bn_plus_identity
        didentity = dz2_bn_plus_identity

        # Backward through second batch norm
        dz2_bn_flat = dz2_bn.reshape(-1, dz2_bn.shape[1])
        dz2_flat, dgamma2, dbeta2 = batch_norm_backward(dz2_bn_flat, self.bn2_cache)
        dz2 = dz2_flat.reshape(self.z2.shape)
        self.d_bn2_gamma = dgamma2.reshape(self.bn2_gamma.shape)
        self.d_bn2_beta = dbeta2.reshape(self.bn2_beta.shape)

        # Backward through second conv
        da1, dw2, db2 = self.conv2d_backward_with_stride(
            self.a1, self.conv2_w, dz2, stride=1, padding=1
        )
        self.d_conv2_w = dw2
        self.d_conv2_b = db2

        # Backward through first ReLU
        dz1_bn = da1 * relu_deriv(self.z1_bn)

        # Backward through first batch norm
        dz1_bn_flat = dz1_bn.reshape(-1, dz1_bn.shape[1])
        dz1_flat, dgamma1, dbeta1 = batch_norm_backward(dz1_bn_flat, self.bn1_cache)
        dz1 = dz1_flat.reshape(self.z1.shape)
        self.d_bn1_gamma = dgamma1.reshape(self.bn1_gamma.shape)
        self.d_bn1_beta = dbeta1.reshape(self.bn1_beta.shape)

        # Backward through first conv
        dx_main, dw1, db1 = self.conv2d_backward_with_stride(
            self.x, self.conv1_w, dz1, stride=self.stride, padding=1
        )
        self.d_conv1_w = dw1
        self.d_conv1_b = db1

        # Backward through shortcut if needed
        if self.downsample:
            # Backward through shortcut batch norm
            dshortcut_bn_flat = didentity.reshape(-1, didentity.shape[1])
            dshortcut_flat, dgamma_sc, dbeta_sc = batch_norm_backward(
                dshortcut_bn_flat, self.shortcut_bn_cache
            )
            dshortcut = dshortcut_flat.reshape(self.shortcut.shape)
            self.d_shortcut_bn_gamma = dgamma_sc.reshape(self.shortcut_bn_gamma.shape)
            self.d_shortcut_bn_beta = dbeta_sc.reshape(self.shortcut_bn_beta.shape)

            # Backward through shortcut conv
            dx_shortcut, dw_sc, db_sc = self.conv2d_backward_with_stride(
                self.x, self.shortcut_w, dshortcut, stride=self.stride, padding=0
            )
            self.d_shortcut_w = dw_sc
            self.d_shortcut_b = db_sc

            dx = dx_main + dx_shortcut
        else:
            dx = dx_main + didentity

        return dx


class ResNet32:
    """ResNet-32 implementation for CIFAR-10."""

    def __init__(
        self,
        num_classes=10,
        dropout_rate=0.5,
        optimizer="sgd",
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
    ):
        self.name = "ResNet32"
        self.dropout_rate = dropout_rate
        self.training = True
        self.optimizer_type = optimizer

        # Initial convolution: 3x32x32 -> 16x32x32
        self.init_conv_w = xp.random.randn(16, 3, 3, 3).astype(xp.float32) * xp.sqrt(
            2.0 / (3 * 3 * 3)
        )
        self.init_conv_b = xp.zeros(16, dtype=xp.float32)
        self.init_bn_gamma = xp.ones((1, 16, 1, 1), dtype=xp.float32)
        self.init_bn_beta = xp.zeros((1, 16, 1, 1), dtype=xp.float32)

        # Residual blocks
        self.blocks = []

        # Stage 1: 16 channels, 5 blocks
        self.blocks.append(ResidualBlock(16, 16, stride=1, downsample=False))
        for _ in range(4):
            self.blocks.append(ResidualBlock(16, 16, stride=1, downsample=False))

        # Stage 2: 32 channels, 5 blocks (first block downsamples)
        self.blocks.append(ResidualBlock(16, 32, stride=2, downsample=True))
        for _ in range(4):
            self.blocks.append(ResidualBlock(32, 32, stride=1, downsample=False))

        # Stage 3: 64 channels, 5 blocks (first block downsamples)
        self.blocks.append(ResidualBlock(32, 64, stride=2, downsample=True))
        for _ in range(4):
            self.blocks.append(ResidualBlock(64, 64, stride=1, downsample=False))

        # Final classification layer: 64 -> num_classes
        self.fc_w = xp.random.randn(64, num_classes).astype(xp.float32) * xp.sqrt(
            2.0 / 64
        )
        self.fc_b = xp.zeros(num_classes, dtype=xp.float32)

        # Initialize gradients
        self.zero_grad()

        # Initialize optimizer
        if optimizer.lower() == "adam":
            self.optimizer = AdamOptimizer(lr=lr, beta1=beta1, beta2=beta2)
        else:
            self.optimizer = None
            self.lr = lr

    def zero_grad(self):
        self.d_init_conv_w = xp.zeros_like(self.init_conv_w)
        self.d_init_conv_b = xp.zeros_like(self.init_conv_b)
        self.d_init_bn_gamma = xp.zeros_like(self.init_bn_gamma)
        self.d_init_bn_beta = xp.zeros_like(self.init_bn_beta)
        self.d_fc_w = xp.zeros_like(self.fc_w)
        self.d_fc_b = xp.zeros_like(self.fc_b)

        for block in self.blocks:
            block.zero_grad()

    def conv2d_with_padding(self, x, w, b, padding=1):
        """Helper for initial convolution with padding."""
        if padding > 0:
            x = xp.pad(
                x,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode="constant",
            )

        N, C, H, W = x.shape
        F, _, k, _ = w.shape
        out_h = H - k + 1
        out_w = W - k + 1

        col = im2col(x, k)
        w_col = w.reshape(F, -1).T
        out = col @ w_col + b
        out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)
        return out

    def conv2d_backward_with_padding(self, x, w, dout, padding=1):
        """Helper for initial convolution backward with padding."""
        if padding > 0:
            x_padded = xp.pad(
                x,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode="constant",
            )
        else:
            x_padded = x

        N, C, H_pad, W_pad = x_padded.shape
        F, _, k, _ = w.shape
        _, _, out_h, out_w = dout.shape

        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, F)
        col = im2col(x_padded, k)

        dw = (col.T @ dout_reshaped).T.reshape(F, C, k, k)
        db = xp.sum(dout_reshaped, axis=0)

        w_col = w.reshape(F, -1)
        dx_col = dout_reshaped @ w_col
        dx_padded = col2im(dx_col, x_padded.shape, k)

        if padding > 0:
            dx = dx_padded[:, :, padding:-padding, padding:-padding]
        else:
            dx = dx_padded

        return dx, dw, db

    def forward(self, x):
        self.x = x

        # Initial convolution + batch norm + ReLU
        self.init_conv = self.conv2d_with_padding(
            x, self.init_conv_w, self.init_conv_b, padding=1
        )
        self.init_bn, self.init_bn_cache = batch_norm_forward(
            self.init_conv.reshape(-1, self.init_conv.shape[1]),
            self.init_bn_gamma.reshape(-1),
            self.init_bn_beta.reshape(-1),
        )
        self.init_bn = self.init_bn.reshape(self.init_conv.shape)
        self.init_relu = relu(self.init_bn)

        # Pass through residual blocks
        current = self.init_relu
        for block in self.blocks:
            current = block.forward(current)

        # Global average pooling
        self.global_pool = xp.mean(current, axis=(2, 3))  # (N, 64)

        # Dropout
        self.pool_drop, self.dropout_mask = dropout_forward(
            self.global_pool, self.dropout_rate, self.training
        )

        # Final classification
        self.fc_out = self.pool_drop @ self.fc_w + self.fc_b
        self.out = softmax(self.fc_out)

        return self.out

    def backward(self, y):
        # Cross-entropy gradient
        dout = cross_entropy_grad(self.out, y)

        # FC layer
        self.d_fc_w = self.pool_drop.T @ dout
        self.d_fc_b = xp.sum(dout, axis=0)
        dpool_drop = dout @ self.fc_w.T

        # Dropout backward
        dglobal_pool = dropout_backward(
            dpool_drop, self.dropout_mask, self.dropout_rate
        )

        # Global average pooling backward
        # Gradient needs to be broadcast back to spatial dimensions
        dcurrent = xp.broadcast_to(
            dglobal_pool[:, :, None, None],
            (dglobal_pool.shape[0], dglobal_pool.shape[1], 8, 8),  # Final spatial size
        ) / (8 * 8)

        # Backward through residual blocks
        for block in reversed(self.blocks):
            dcurrent = block.backward(dcurrent)

        # Backward through initial layers
        dinit_relu = dcurrent * relu_deriv(self.init_bn)

        # Initial batch norm backward
        dinit_bn_flat = dinit_relu.reshape(-1, dinit_relu.shape[1])
        dinit_conv_flat, dgamma_init, dbeta_init = batch_norm_backward(
            dinit_bn_flat, self.init_bn_cache
        )
        dinit_conv = dinit_conv_flat.reshape(self.init_conv.shape)
        self.d_init_bn_gamma = dgamma_init.reshape(self.init_bn_gamma.shape)
        self.d_init_bn_beta = dbeta_init.reshape(self.init_bn_beta.shape)

        # Initial conv backward
        dx, dw_init, db_init = self.conv2d_backward_with_padding(
            self.x, self.init_conv_w, dinit_conv, padding=1
        )
        self.d_init_conv_w = dw_init
        self.d_init_conv_b = db_init

    def step(self, lr=None):
        """Update parameters using the specified optimizer."""
        if self.optimizer_type.lower() == "adam":
            self.optimizer.step()
            # Update initial layer parameters
            self.init_conv_w = self.optimizer.update(
                "init_conv_w", self.init_conv_w, self.d_init_conv_w
            )
            self.init_conv_b = self.optimizer.update(
                "init_conv_b", self.init_conv_b, self.d_init_conv_b
            )
            self.init_bn_gamma = self.optimizer.update(
                "init_bn_gamma", self.init_bn_gamma, self.d_init_bn_gamma
            )
            self.init_bn_beta = self.optimizer.update(
                "init_bn_beta", self.init_bn_beta, self.d_init_bn_beta
            )
            self.fc_w = self.optimizer.update("fc_w", self.fc_w, self.d_fc_w)
            self.fc_b = self.optimizer.update("fc_b", self.fc_b, self.d_fc_b)

            # Update residual block parameters
            for i, block in enumerate(self.blocks):
                prefix = f"block_{i}_"
                block.conv1_w = self.optimizer.update(
                    prefix + "conv1_w", block.conv1_w, block.d_conv1_w
                )
                block.conv1_b = self.optimizer.update(
                    prefix + "conv1_b", block.conv1_b, block.d_conv1_b
                )
                block.bn1_gamma = self.optimizer.update(
                    prefix + "bn1_gamma", block.bn1_gamma, block.d_bn1_gamma
                )
                block.bn1_beta = self.optimizer.update(
                    prefix + "bn1_beta", block.bn1_beta, block.d_bn1_beta
                )
                block.conv2_w = self.optimizer.update(
                    prefix + "conv2_w", block.conv2_w, block.d_conv2_w
                )
                block.conv2_b = self.optimizer.update(
                    prefix + "conv2_b", block.conv2_b, block.d_conv2_b
                )
                block.bn2_gamma = self.optimizer.update(
                    prefix + "bn2_gamma", block.bn2_gamma, block.d_bn2_gamma
                )
                block.bn2_beta = self.optimizer.update(
                    prefix + "bn2_beta", block.bn2_beta, block.d_bn2_beta
                )

                if block.downsample:
                    block.shortcut_w = self.optimizer.update(
                        prefix + "shortcut_w", block.shortcut_w, block.d_shortcut_w
                    )
                    block.shortcut_b = self.optimizer.update(
                        prefix + "shortcut_b", block.shortcut_b, block.d_shortcut_b
                    )
                    block.shortcut_bn_gamma = self.optimizer.update(
                        prefix + "shortcut_bn_gamma",
                        block.shortcut_bn_gamma,
                        block.d_shortcut_bn_gamma,
                    )
                    block.shortcut_bn_beta = self.optimizer.update(
                        prefix + "shortcut_bn_beta",
                        block.shortcut_bn_beta,
                        block.d_shortcut_bn_beta,
                    )
        else:
            # SGD update
            effective_lr = lr if lr is not None else self.lr

            # Update initial layer parameters
            self.init_conv_w -= effective_lr * self.d_init_conv_w
            self.init_conv_b -= effective_lr * self.d_init_conv_b
            self.init_bn_gamma -= effective_lr * self.d_init_bn_gamma
            self.init_bn_beta -= effective_lr * self.d_init_bn_beta
            self.fc_w -= effective_lr * self.d_fc_w
            self.fc_b -= effective_lr * self.d_fc_b

            # Update residual block parameters
            for block in self.blocks:
                block.conv1_w -= effective_lr * block.d_conv1_w
                block.conv1_b -= effective_lr * block.d_conv1_b
                block.bn1_gamma -= effective_lr * block.d_bn1_gamma
                block.bn1_beta -= effective_lr * block.d_bn1_beta
                block.conv2_w -= effective_lr * block.d_conv2_w
                block.conv2_b -= effective_lr * block.d_conv2_b
                block.bn2_gamma -= effective_lr * block.d_bn2_gamma
                block.bn2_beta -= effective_lr * block.d_bn2_beta

                if block.downsample:
                    block.shortcut_w -= effective_lr * block.d_shortcut_w
                    block.shortcut_b -= effective_lr * block.d_shortcut_b
                    block.shortcut_bn_gamma -= effective_lr * block.d_shortcut_bn_gamma
                    block.shortcut_bn_beta -= effective_lr * block.d_shortcut_bn_beta

    def train(self):
        """Set model to training mode."""
        self.training = True

    def eval(self):
        """Set model to evaluation mode."""
        self.training = False
