import torch
from .pde_solver import solve_state, solve_adjoint, u_star
from .ControlVector import ControlVector
from .sampling import interior_mask_2d, build_fd_laplacian_2d
class ReducedSemilinearControlObjective:
    """
    Smooth reduced objective:
        j_sm(u) = 0.5 ||y(u)-g_d||^2 + 0.5 alpha ||u||^2

    where y(u) solves
        -Delta y + max(0,y) = u + f,  y=0 on boundary.
    """

    def __init__(
        self,
        xy,
        g_d,
        y_true,
        f_rhs,
        alpha=1e-2,
        ngrid=32,
        weight=None,
        device="cpu",
        mu_I=0.0,
        newton_tol=1e-10,
        newton_maxit=30,
        fd_eps=1e-6,
    ):
        self.xy = xy.to(device)
        self.g_d = g_d.to(device)
        self.y_true = y_true.to(device)
        self.f_rhs = f_rhs.to(device)
        self.alpha = float(alpha)
        self.ngrid = int(ngrid)
        self.device = device
        self.mu_I = float(mu_I)
        self.newton_tol = float(newton_tol)
        self.newton_maxit = int(newton_maxit)
        self.fd_eps = float(fd_eps)
        self.hess_mode = "full"

        h = 1.0 / (ngrid - 1)
        if weight is None:
            self.weight = torch.tensor(h * h, dtype=self.xy.dtype, device=device)
        else:
            self.weight = weight.to(device)

        self.mask = interior_mask_2d(ngrid, device=device)
        self.xy_int = self.xy[self.mask]
        self.g_int = self.g_d[self.mask]
        self.y_true_int = self.y_true[self.mask]
        self.f_int = self.f_rhs[self.mask]
        self.A = build_fd_laplacian_2d(ngrid, h, device=device, dtype=self.xy.dtype)
        self._last_y = None

    def set_mu_I(self, mu_I: float):
        self.mu_I = float(mu_I)

    def set_hess_mode(self, mode: str):
        self.hess_mode = mode

    def update(self, x, flag: str):
        pass

    def solve_state_cached(self, u: ControlVector):
        y0 = torch.zeros_like(u.data) if self._last_y is None else self._last_y.clone()
        y = solve_state(
            u.data, self.f_int, y0, self.A,
            newton_tol=self.newton_tol,
            newton_maxit=self.newton_maxit
        )
        self._last_y = y.detach().clone()
        return y

    def value(self, x, ftol=1e-12):
        y = self.solve_state_cached(x)
        w = self.weight
        J_state = 0.5 * w * torch.sum((y - self.g_int) ** 2)
        J_ctrl = 0.5 * self.alpha * w * torch.sum(x.data ** 2)
        J = J_state + J_ctrl
        return float(J.detach().cpu().item()), 0.0

    def gradient(self, x, gtol=1e-12):
        y = self.solve_state_cached(x)
        p = solve_adjoint(y, self.g_int, self.A, self.weight)
        grad = self.alpha * self.weight * x.data - p
        return ControlVector(grad.detach().clone()), 0.0

    def hessVec(self, v, x, gradTol=1e-12):
        eps = self.fd_eps

        xp = x.copy()
        xm = x.copy()
        xp.axpy(eps, v)
        xm.axpy(-eps, v)

        gp, _ = self.gradient(xp)
        gm, _ = self.gradient(xm)

        hv = gp.copy()
        hv.axpy(-1.0, gm)
        hv.scal(1.0 / (2.0 * eps))

        if self.mu_I != 0.0:
            hv.axpy(self.mu_I, v)

        return hv, 0.0

    def relative_L2_error_control(self, x):
        u_ex = u_star(self.xy_int, self.alpha)
        err = torch.sqrt(torch.sum((x.data - u_ex) ** 2) / torch.sum(u_ex ** 2))
        return float(err.item())

    def relative_L2_error_state(self, x):
        y = self.solve_state_cached(x)
        y_ex = self.y_true_int
        err = torch.sqrt(torch.sum((y - y_ex) ** 2) / torch.sum(y_ex ** 2))
        return float(err.item())
