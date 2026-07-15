import torch

def make_interior_grid(ngrid, device="cpu", dtype=torch.float64):
    grid = torch.linspace(0.0, 1.0, ngrid, dtype=dtype, device=device)
    interior = grid[1:-1]
    X, Y = torch.meshgrid(interior, interior, indexing="ij")
    xy = torch.stack((X.reshape(-1), Y.reshape(-1)), dim=1)
    return grid, X, Y, xy

def desired_state(xy):
    x1, x2 = xy[:, 0:1], xy[:, 1:2]
    return (0.5*torch.sin(torch.pi*x1)*torch.sin(torch.pi*x2)
            +0.25*torch.exp(-70*((x1-0.35)**2+(x2-0.65)**2))
            -0.20*torch.exp(-90*((x1-0.70)**2+(x2-0.30)**2)))

def build_negative_laplacian(ngrid, device="cpu", dtype=torch.float64):
    n = ngrid - 2
    h = 1.0/(ngrid-1)
    main = 2.0*torch.ones(n, dtype=dtype, device=device)
    off = -torch.ones(n-1, dtype=dtype, device=device)
    T = torch.diag(main)+torch.diag(off,1)+torch.diag(off,-1)
    I = torch.eye(n, dtype=dtype, device=device)
    A = (torch.kron(I,T)+torch.kron(T,I))/(h*h)
    return A, h
