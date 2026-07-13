from .ControlVector import ControlVector
from .Reduced_obj import ReducedSemilinearControlObjective
from .pde_solver_smoothing import (
    solve_adjoint_smooth,
    solve_state_smooth,
)


class SmoothedReducedSemilinearControlObjective(
    ReducedSemilinearControlObjective
):
    """
    Reduced objective equipped with a smoothing model for the PDE
    nonlinearity.

    The inherited value(), gradient(), and hessVec() methods correspond
    to the original semismooth problem.

    The methods value_smooth(), gradient_smooth(), and
    hessVec_smooth() correspond to the Huber-smoothed PDE.
    """

    def solve_state_smooth(self, x: ControlVector, mu: float):
        # Avoid reusing a cached state computed with a different mu.
        y0 = x.data.new_zeros(x.data.shape)

        return solve_state_smooth(
            x.data,
            self.f_int,
            y0,
            self.A,
            mu=mu,
            newton_tol=self.newton_tol,
            newton_maxit=self.newton_maxit,
        )

    def value_smooth(self, x, mu, ftol=1e-12):
        y = self.solve_state_smooth(x, mu)
        w = self.weight

        state_term = 0.5 * w * ((y - self.g_int) ** 2).sum()
        control_term = 0.5 * self.alpha * w * (x.data ** 2).sum()

        value = state_term + control_term
        return float(value.detach().cpu().item()), 0.0

    def gradient_smooth(self, x, mu, gtol=1e-12):
        y = self.solve_state_smooth(x, mu)

        p = solve_adjoint_smooth(
            y,
            self.g_int,
            self.A,
            self.weight,
            mu,
        )

        gradient = self.alpha * self.weight * x.data - p
        return ControlVector(gradient.detach().clone()), 0.0

    def hessVec_smooth(self, v, x, mu, gradTol=1e-12):
        """
        Finite-difference Hessian-vector product of the smoothed
        reduced gradient.
        """
        eps = self.fd_eps

        xp = x.copy()
        xm = x.copy()
        xp.axpy(eps, v)
        xm.axpy(-eps, v)

        gp, _ = self.gradient_smooth(xp, mu, gradTol)
        gm, _ = self.gradient_smooth(xm, mu, gradTol)

        hv = gp.copy()
        hv.axpy(-1.0, gm)
        hv.scal(1.0 / (2.0 * eps))

        if self.mu_I != 0.0:
            hv.axpy(self.mu_I, v)

        return hv, 0.0