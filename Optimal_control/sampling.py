import torch

def make_training_points_grid(n=32, device="cpu", dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    xs = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return X, Y, xy


def interior_mask_2d(n, device="cpu"):
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    mask[1:-1, 1:-1] = True
    return mask.reshape(-1)


def build_fd_laplacian_2d(n, h, device="cpu", dtype=torch.float64):
    """
    Matrix for -Delta on the interior (n-2)x(n-2) grid.
    """
    ni = n - 2
    m = ni * ni
    A = torch.zeros((m, m), dtype=dtype, device=device)

    def idx(i, j):
        return i * ni + j

    c = 1.0 / (h * h)
    for i in range(ni):
        for j in range(ni):
            k = idx(i, j)
            A[k, k] = 4.0 * c
            if i > 0:
                A[k, idx(i - 1, j)] = -c
            if i < ni - 1:
                A[k, idx(i + 1, j)] = -c
            if j > 0:
                A[k, idx(i, j - 1)] = -c
            if j < ni - 1:
                A[k, idx(i, j + 1)] = -c
    return A


