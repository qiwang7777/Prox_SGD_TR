import torch

from .GridVector import GridVector


class SparseReLUOperatorObjective:
    """
    Semismooth objective

        f(u) = 0.5 ||ReLU(Ku) - g||_{L2}^2
               + 0.5 * alpha ||u||_{L2}^2.

    The Gaussian integral operator is discretized on a tensor-product grid.
    Since the kernel is separable,

        exp(-|x-z|^2 / (2 sigma^2))
        = exp(-(x1-z1)^2 / (2 sigma^2))
          exp(-(x2-z2)^2 / (2 sigma^2)),

    K is applied as

        K(U) = weight * G @ U @ G.T,

    avoiding formation of a dense ngrid^2 by ngrid^2 matrix.

    At points where Ku = 0, a Clarke selection theta in [0,1] is used.
    The policy is controlled by relu_zero_selection:
        "inactive" -> theta = 0
        "midpoint" -> theta = 0.5
        "active"   -> theta = 1
    """

    def __init__(
        self,
        xy,
        g,
        alpha=1e-4,
        sigma=0.08,
        ngrid=64,
        weight=None,
        device="cpu",
        relu_zero_selection="midpoint",
        zero_tol=1e-12,
        fd_eps=1e-6,
    ):
        self.xy = xy.to(device)
        self.g = g.to(device)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.ngrid = int(ngrid)
        self.device = device
        self.relu_zero_selection = str(relu_zero_selection).lower()
        self.zero_tol = float(zero_tol)
        self.fd_eps = float(fd_eps)

        h = 1.0 / (self.ngrid - 1)
        self.weight = float(h * h) if weight is None else float(weight)

        grid = torch.linspace(
            0.0,
            1.0,
            self.ngrid,
            dtype=self.xy.dtype,
            device=device,
        )
        diff = grid[:, None] - grid[None, :]
        self.G = torch.exp(-(diff * diff) / (2.0 * self.sigma * self.sigma))

        self.hess_mode = "full"
        self.x_true = None
        self._last_Ku = None

    @torch.no_grad()
    def relative_L2_error(self, x):
        """
        Compatibility hook used by semismooth_TR.trust_region.

        If no reference solution is supplied, return infinity.  If x_true is
        a GridVector or tensor, compute the weighted relative L2 error.
        """
        if self.x_true is None:
            return float("inf")

        true_data = (
            self.x_true.data
            if hasattr(self.x_true, "data")
            else self.x_true
        )
        diff = x.data - true_data
        numerator = torch.sqrt(self.weight * torch.sum(diff * diff))
        denominator = torch.sqrt(self.weight * torch.sum(true_data * true_data))

        denom = float(denominator.item())
        if denom == 0.0:
            return float(numerator.item())
        return float((numerator / denominator).item())


    def set_mu_I(self, mu_I: float):
        # Kept for compatibility with the existing TR code.
        self.mu_I = float(mu_I)

    def set_hess_mode(self, mode: str):
        self.hess_mode = mode

    def update(self, x, flag: str):
        # No state equation cache needs to be accepted/rejected.
        pass

    def _reshape(self, x):
        return x.data.reshape(self.ngrid, self.ngrid)

    def apply_K_tensor(self, u_tensor):
        return self.weight * (self.G @ u_tensor @ self.G.T)

    def apply_K(self, x):
        Ku = self.apply_K_tensor(self._reshape(x))
        return Ku.reshape(-1, 1)

    def apply_K_adjoint_tensor(self, v_tensor):
        # The Gaussian kernel and quadrature discretization are symmetric.
        return self.weight * (self.G.T @ v_tensor @ self.G)

    def apply_K_adjoint(self, v):
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        V = v.reshape(self.ngrid, self.ngrid)
        return self.apply_K_adjoint_tensor(V).reshape(-1, 1)

    def _theta(self, Ku):
        theta = torch.zeros_like(Ku)
        theta[Ku > self.zero_tol] = 1.0

        zero_mask = torch.abs(Ku) <= self.zero_tol
        if self.relu_zero_selection == "inactive":
            theta[zero_mask] = 0.0
        elif self.relu_zero_selection == "active":
            theta[zero_mask] = 1.0
        elif self.relu_zero_selection == "midpoint":
            theta[zero_mask] = 0.5
        else:
            raise ValueError(
                "relu_zero_selection must be 'inactive', 'midpoint', or 'active'"
            )
        return theta

    def value(self, x, ftol=1e-12):
        Ku = self.apply_K(x)
        self._last_Ku = Ku.detach().clone()

        residual = torch.relu(Ku) - self.g
        state_term = 0.5 * self.weight * torch.sum(residual * residual)
        reg_term = 0.5 * self.alpha * self.weight * torch.sum(x.data * x.data)
        value = state_term + reg_term
        return float(value.detach().cpu().item()), 0.0

    def value_model(self, x, ftol=1e-12):
        return self.value(x, ftol)

    def gradient(self, x, gtol=1e-12):
        """
        Return one Clarke generalized gradient.

        For z = Ku, the scalar loss is
            q(z) = 0.5 * (ReLU(z) - g)^2.
        A Clarke derivative selection is
            theta * (ReLU(z) - g),  theta in partial ReLU(z).
        """
        Ku = self.apply_K(x)
        self._last_Ku = Ku.detach().clone()

        residual = torch.relu(Ku) - self.g
        theta = self._theta(Ku)
        chain = theta * residual

        grad_data = (
            self.weight * self.apply_K_adjoint(chain)
            + self.alpha * self.weight * x.data
        )
        return GridVector(grad_data.detach().clone()), 0.0

    def hessVec(self, v, x, gradTol=1e-12):
        """
        Finite-difference generalized-gradient action.

        SPG2 is the recommended subproblem solver for this example, so this
        routine is mainly supplied for compatibility with NCG/SSN experiments.
        """
        #eps = self.fd_eps
        #xp = x.copy()
        #xm = x.copy()
        #xp.axpy(eps, v)
        #xm.axpy(-eps, v)

        #gp, _ = self.gradient(xp, gradTol)
        #gm, _ = self.gradient(xm, gradTol)
        #hv = gp.copy()
        #hv.axpy(-1.0, gm)
        #hv.scal(1.0 / (2.0 * eps))
        #return hv, 0.0
        return v.zero_like(), 0.0

    @torch.no_grad()
    def relu_active_fraction(self, x):
        Ku = self.apply_K(x)
        return float(torch.mean((Ku > self.zero_tol).to(Ku.dtype)).item())

    @torch.no_grad()
    def relu_kink_fraction(self, x):
        Ku = self.apply_K(x)
        return float(torch.mean((torch.abs(Ku) <= self.zero_tol).to(Ku.dtype)).item())
