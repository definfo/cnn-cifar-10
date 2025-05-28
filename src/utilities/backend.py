try:
    import cupy as cp

    xp = cp

    def to_cpu(arr):
        return cp.asnumpy(arr)

    print("[Backend] Using CuPy (GPU)")
except ImportError:
    import numpy as np

    xp = np

    def to_cpu(arr):
        return arr

    print("[Backend] Using NumPy (CPU)")
