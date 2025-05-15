import numpy as np

def remove_diag_reshape(matrix):
    matrix = np.copy(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("must be n*n matrix")
    
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    flattened = matrix[mask]
    return flattened.reshape(n, n - 1)

def reconstruct_with_diagonal(flattened, diag_value=0):
    flattened = np.copy(flattened)
    length = len(flattened)
    n = int((1 + np.sqrt(1 + 4 * length)) / 2)
    
    if n * (n - 1) != length:
        raise ValueError("input error")

    matrix = np.empty((n, n), dtype=flattened.dtype)
    mask = ~np.eye(n, dtype=bool)
    matrix[mask] = flattened
    np.fill_diagonal(matrix, diag_value)
    return matrix

def reconstruct(flattened):
    flattened = np.copy(flattened)
    length = len(flattened)
    n = int(np.sqrt(length))   
    matrix = flattened.reshape((n,n))
    return matrix

def contrastive_normalization_np(sim):
    sim = np.copy(sim)
    col_min = sim.min(axis=0, keepdims=True)

    col_max = sim.max(axis=0, keepdims=True)
    
    # col_max = sim.max(axis=0, keepdims=True)

    eps = 1e-8
    normalized = (sim - col_min) / (col_max - col_min + eps)
    normalized = np.clip(normalized, 0, 1)
    # normalized = (normalized + normalized.T) / 2
    normalized = np.power(normalized, 2)
   
    return normalized