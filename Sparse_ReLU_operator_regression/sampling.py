import torch


def make_grid(ngrid, device="cpu", dtype=torch.float64):
    grid = torch.linspace(0.0, 1.0, ngrid, device=device, dtype=dtype)
    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    xy = torch.stack((X.reshape(-1), Y.reshape(-1)), dim=1)
    return X, Y, xy


def target_function(xy):
    x1 = xy[:, 0:1]
    x2 = xy[:, 1:2]

    latent = (
        0.6 * torch.sin(2.0 * torch.pi * x1) * torch.sin(torch.pi * x2)
        + 0.4 * torch.exp(
            -60.0 * ((x1 - 0.35) ** 2 + (x2 - 0.65) ** 2)
        )
        - 0.3 * torch.exp(
            -80.0 * ((x1 - 0.70) ** 2 + (x2 - 0.30) ** 2)
        )
    )
    return torch.relu(latent)
