import torch

from .pde_solver import (
    desired_state,
    f_source,
    laplacian_of_function,
    p_star,
    u_star,
    y_star,
)


import torch
import torch.nn.functional as F


def eval_N_and_dNdy_smooth(y: torch.Tensor, mu: float):
    """
    Smooth approximation of N(y) = ReLU(y):

        N_mu(y) = mu * log(1 + exp(y / mu)),
        N_mu'(y) = sigmoid(y / mu).
    """
    if mu <= 0.0:
        raise ValueError("mu must be positive.")

    mu_t = torch.as_tensor(mu, dtype=y.dtype, device=y.device)

    # Numerically stable softplus implementation
    Nval = mu_t * F.softplus(y / mu_t)
    dNdy = torch.sigmoid(y / mu_t)

    return Nval, dNdy


def solve_state_smooth(
    u_int,
    f_int,
    y0,
    A,
    mu,
    newton_tol=1e-10,
    newton_maxit=30,
):
    """
    Solve the smoothed state equation

        A y + N_mu(y) - u - f = 0.
    """
    y = y0.clone()

    for _ in range(newton_maxit):
        Nval, dNdy = eval_N_and_dNdy_smooth(y, mu)
        residual = A @ y + Nval - u_int - f_int

        if torch.linalg.norm(residual).item() < newton_tol:
            return y

        jacobian = A + torch.diag(dNdy.reshape(-1))
        step = torch.linalg.solve(jacobian, -residual)
        y = y + step

        if torch.linalg.norm(step).item() < newton_tol:
            return y

    raise RuntimeError("Smoothed state Newton solver did not converge.")


def solve_adjoint_smooth(y_int, g_int, A, weight, mu):
    """
    Solve the adjoint equation associated with the smoothed PDE:

        (A + diag(N_mu'(y)))^T p
            = -weight * (y - g).
    """
    _, dNdy = eval_N_and_dNdy_smooth(y_int, mu)

    jacobian = A + torch.diag(dNdy.reshape(-1))
    rhs = -weight * (y_int - g_int)

    return torch.linalg.solve(jacobian.T, rhs)
