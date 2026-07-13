import torch

from .pde_solver import (
    desired_state,
    f_source,
    laplacian_of_function,
    p_star,
    u_star,
    y_star,
)


def eval_N_and_dNdy_smooth(y: torch.Tensor, mu: float):
    """
    Huber smoothing of N(y) = max(0, y).

    N_mu(y) =
        0,                  y <= 0,
        y^2 / (2 mu),       0 < y < mu,
        y - mu / 2,         y >= mu.

    N_mu'(y) = clip(y / mu, 0, 1).
    """
    if mu <= 0.0:
        raise ValueError("The smoothing parameter mu must be positive.")

    mu_t = torch.as_tensor(mu, dtype=y.dtype, device=y.device)

    Nval = torch.where(
        y <= 0.0,
        torch.zeros_like(y),
        torch.where(
            y < mu_t,
            y.square() / (2.0 * mu_t),
            y - 0.5 * mu_t,
        ),
    )

    dNdy = torch.clamp(y / mu_t, min=0.0, max=1.0)
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
