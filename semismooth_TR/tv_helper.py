import torch

def grad2d(u):
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)

    gx[:-1, :] = u[1:, :] - u[:-1, :]
    gy[:, :-1] = u[:, 1:] - u[:, :-1]
    return gx, gy


def div2d(px, py):
    out = torch.zeros_like(px)

    out[:-1, :] -= px[:-1, :]
    out[1:,  :] += px[:-1, :]

    out[:, :-1] -= py[:, :-1]
    out[:, 1: ] += py[:, :-1]

    return out


def tv_value_isotropic(u):
    gx, gy = grad2d(u)
    return torch.sum(torch.sqrt(gx * gx + gy * gy)) #+ 1e-16))


@torch.no_grad()
def prox_tv_chambolle(v, weight, max_iter=100, tol=1e-5):
    """
    Computes prox_{weight * TV}(v) for 2D isotropic TV:
        argmin_u 0.5||u-v||^2 + weight*TV(u)
    """
    if weight <= 0:
        return v.detach().clone()

    px = torch.zeros_like(v)
    py = torch.zeros_like(v)

    tau = 0.25

    for _ in range(max_iter):
        px_old = px.clone()
        py_old = py.clone()

        div_p = div2d(px, py)
        gx, gy = grad2d(div_p - v / weight)

        px = px + tau * gx
        py = py + tau * gy

        norm = torch.maximum(torch.ones_like(v), torch.sqrt(px * px + py * py))
        px = px / norm
        py = py / norm

        err = max((px - px_old).abs().max().item(), (py - py_old).abs().max().item())
        if err < tol:
            break

    return v - weight * div2d(px, py)
