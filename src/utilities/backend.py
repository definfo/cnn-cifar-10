try:
    import cupy as cp

    xp = cp
    HAS_CUPY = True

    def to_cpu(arr):
        if hasattr(arr, "get"):
            # If the array has a 'get' method, it's a CuPy array on GPU
            return arr.get()
        return arr

    print("[Backend] Using CuPy (GPU)")
except (ModuleNotFoundError, ImportError):
    import numpy as np

    xp = np
    HAS_CUPY = False

    def to_cpu(arr):
        return arr

    print("[Backend] Using NumPy (CPU)")


def convert_array_to_current_backend(arr):
    """Convert array to current backend (NumPy or CuPy)"""
    # Handle None or non-array values
    if arr is None or not hasattr(arr, "shape"):
        return arr

    if HAS_CUPY:
        # If we have CuPy, convert to CuPy array
        if hasattr(arr, "get"):
            # Already a CuPy array
            return arr
        else:
            # NumPy array, convert to CuPy
            return cp.array(arr)
    else:
        # If we don't have CuPy, ensure it's a NumPy array
        if hasattr(arr, "get"):
            # CuPy array, convert to NumPy
            return arr.get()
        else:
            # Already NumPy array or compatible, ensure it's NumPy
            if not isinstance(arr, np.ndarray):
                return np.array(arr)
            return arr
