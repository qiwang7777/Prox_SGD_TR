import torch
import math
# ============================================================
# Exact example from the paper
#    y = p =
#      [ ((x1-1/2)^4 + 1/2 (x1-1/2)^3) sin(pi x2),   x1 < 1/2
#      [ 0,                                           x1 >= 1/2
#
#    PDE:
#      -Delta y + max(0,y) = u + f
# ============================================================
def y_star(xy: torch.Tensor) -> torch.Tensor:
    x1 = xy[:, 0:1]
    x2 = xy[:, 1:2]
    left = ((x1 - 0.5) ** 4 + 0.5 * (x1 - 0.5) ** 3) * torch.sin(math.pi * x2)
    return torch.where(x1 < 0.5, left, torch.zeros_like(left))


def p_star(xy: torch.Tensor) -> torch.Tensor:
    return y_star(xy)


def u_star(xy: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    From the first-order condition: alpha * u + p = 0
    """
    return (1.0 / alpha) * p_star(xy)

def chi_y_positive(xy:torch.Tensor):
    y = y_star(xy)
    return (y>0).to(y.dtype)


def laplacian_of_function(fun, xy: torch.Tensor) -> torch.Tensor:
    xy_req = xy.detach().clone().requires_grad_(True)
    val = fun(xy_req)

    grad_val = torch.autograd.grad(val.sum(), xy_req, create_graph=True)[0]

    lap = 0.0
    for j in range(xy.shape[1]):
        second_j = torch.autograd.grad(
            grad_val[:, j].sum(), xy_req, create_graph=True
        )[0][:, j:j+1]
        lap = lap + second_j

    return lap.detach()

def desired_state(xy:torch.Tensor):
    y = y_star(xy)
    p = p_star(xy)
    lap_p = laplacian_of_function(p_star,xy)
    chi = (y>0).to(y.dtype)
    return y-lap_p+chi*p
    

def f_source(xy: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute f from the exact PDE:
        -Delta y + max(0,y) = u + f
    so
        f = -Delta y + max(0,y) - u
    """
    y = y_star(xy)
    lap_y = laplacian_of_function(y_star, xy)
    u = u_star(xy, alpha)
    return -lap_y + torch.relu(y) - u

# ============================================================
#  Semismooth nonlinearity max(0,y)
# ============================================================

def eval_N_and_dNdy(y):
    """
    N(y) = max(0,y), and one generalized derivative selection:
        dN/dy = 1 if y>0, 0 if y<=0
    """
    Nval = torch.relu(y)
    dNdy = (y > 0).to(y.dtype)
    return Nval, dNdy


# ============================================================
# 7) PDE solvers
# ============================================================

def solve_state(u_int, f_int, y0, A, newton_tol=1e-10, newton_maxit=30):
    """
    Solve:
        -Delta y + max(0,y) = u + f
    i.e.
        A y + max(0,y) - u - f = 0
    """
    y = y0.clone()

    for _ in range(newton_maxit):
        Nval, dNdy = eval_N_and_dNdy(y)
        F = A @ y + Nval - u_int - f_int

        if torch.linalg.norm(F).item() < newton_tol:
            return y

        J = A + torch.diag(dNdy.reshape(-1))
        dy = torch.linalg.solve(J, -F)
        y = y + dy

        if torch.linalg.norm(dy).item() < newton_tol:
            return y

    raise RuntimeError("State Newton solver did not converge.")


def solve_adjoint(y_int, g_int, A, weight):
    """
    Adjoint equation for the discrete reduced objective:
        (A + diag(dN/dy))^T p = - weight * (y - g)
    """
    _, dNdy = eval_N_and_dNdy(y_int)
    J = A + torch.diag(dNdy.reshape(-1))
    rhs = -weight * (y_int - g_int)
    p = torch.linalg.solve(J.T, rhs)
    return p

