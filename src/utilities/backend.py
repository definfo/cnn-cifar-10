try:
    import cupy as cp

    xp = cp

    def to_cpu(arr):
        if hasattr(arr, "get"):
            # If the array has a 'get' method, it's a CuPy array on GPU
            return arr.get()
        return arr

    print("[Backend] Using CuPy (GPU)")
except ImportError:
    import numpy as np

    xp = np

    def to_cpu(arr):
        return arr

    print("[Backend] Using NumPy (CPU)")
