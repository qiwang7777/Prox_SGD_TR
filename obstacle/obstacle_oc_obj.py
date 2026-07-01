import torch


def make_grid(n=64, device="cpu", dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()

    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    h = 1.0 / (n - 1)

    return xy, h


def psi_fun(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return 0.2 * torch.exp(-50.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def yd_fun(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return -0.2 * torch.exp(-50.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def softplus_mu(z, mu):
    return mu * torch.nn.functional.softplus(z / mu)


class BoxL1Norm:
    """
    phi(u) = beta ||u||_1 + indicator_{[umin, umax]}(u)

    prox = projection onto box after soft-thresholding.
    """

    def __init__(self, beta=1e-5, umin=-5.0, umax=5.0):
        self.beta = beta
        self.umin = umin
        self.umax = umax

    def value(self, u):
        in_box = torch.all((u >= self.umin) & (u <= self.umax))
        if not in_box:
            return float("inf")
        return self.beta * torch.sum(torch.abs(u))

    def prox(self, u, r):
        z = torch.sign(u) * torch.clamp(torch.abs(u) - r * self.beta, min=0.0)
        return torch.clamp(z, self.umin, self.umax)


class ObstacleControlObjective:
    """
    f(u) =
        0.5 ||y - yd||^2
      + 0.5 alpha ||u||^2
      + 0.5 gamma ||max(0, psi - y)||^2

    where -Delta y = u, y = 0 on boundary.

    phi(u) is handled separately by BoxL1Norm.
    """

    def __init__(
        self,
        n=64,
        alpha=1e-3,
        gamma=1e2,
        device="cpu",
        dtype=None,
    ):
        if dtype is None:
            dtype = torch.get_default_dtype()

        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.dtype = dtype

        self.xy, self.h = make_grid(n, device=device, dtype=dtype)
        self.weight = self.h ** 2

        self.psi = psi_fun(self.xy)
        self.yd = yd_fun(self.xy)

        self.A = self._build_poisson_matrix().to(device=device, dtype=dtype)

    def _build_poisson_matrix(self):
        n = self.n
        h = self.h
        N = n * n

        A = torch.zeros((N, N), dtype=self.dtype)

        def idx(i, j):
            return i * n + j

        for i in range(n):
            for j in range(n):
                p = idx(i, j)

                if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                    A[p, p] = 1.0
                else:
                    A[p, p] = 4.0 / h ** 2
                    A[p, idx(i - 1, j)] = -1.0 / h ** 2
                    A[p, idx(i + 1, j)] = -1.0 / h ** 2
                    A[p, idx(i, j - 1)] = -1.0 / h ** 2
                    A[p, idx(i, j + 1)] = -1.0 / h ** 2

        return A

    def solve_state(self, u):
        rhs = u.reshape(-1, 1).clone()

        n = self.n
        rhs_grid = rhs.reshape(n, n)

        rhs_grid[0, :] = 0.0
        rhs_grid[-1, :] = 0.0
        rhs_grid[:, 0] = 0.0
        rhs_grid[:, -1] = 0.0

        rhs = rhs_grid.reshape(-1, 1)

        y = torch.linalg.solve(self.A, rhs)
        return y

    def value_torch(self, u):
        y = self.solve_state(u)

        tracking = 0.5 * self.weight * torch.sum((y - self.yd) ** 2)
        control = 0.5 * self.alpha * self.weight * torch.sum(u ** 2)

        violation = self.psi - y
        obstacle = 0.5 * self.gamma * self.weight * torch.sum(
            torch.relu(violation) ** 2
        )

        return tracking + control + obstacle

    def value_smooth_torch(self, u, mu):
        y = self.solve_state(u)

        tracking = 0.5 * self.weight * torch.sum((y - self.yd) ** 2)
        control = 0.5 * self.alpha * self.weight * torch.sum(u ** 2)

        violation = self.psi - y
        smooth_violation = softplus_mu(violation, mu)

        obstacle = 0.5 * self.gamma * self.weight * torch.sum(
            smooth_violation ** 2
        )

        return tracking + control + obstacle

    def value(self, u, tol=1e-12):
        val = self.value_torch(u)
        return float(val.detach().cpu().item()), 0.0

    def gradient(self, u, tol=1e-12):
        u_ad = u.detach().clone().requires_grad_(True)
        val = self.value_torch(u_ad)

        grad = torch.autograd.grad(
            val,
            u_ad,
            create_graph=False,
            retain_graph=False,
        )[0]

        return grad.detach(), 0.0

    def hessvec(self, u, v, tol=1e-12):
        u_ad = u.detach().clone().requires_grad_(True)
        val = self.value_torch(u_ad)

        grad = torch.autograd.grad(
            val,
            u_ad,
            create_graph=True,
            retain_graph=True,
        )[0]

        gv = torch.dot(grad.reshape(-1), v.reshape(-1))

        hv = torch.autograd.grad(
            gv,
            u_ad,
            retain_graph=False,
        )[0]

        return hv.detach(), 0.0

    def attach_smoother(self, smoother):
        self.smoother = smoother

    def value_smooth(self, u, mu, tol=1e-12):
        return self.smoother.value(u, mu, tol)

    def gradient_smooth(self, u, mu, tol=1e-12):
        return self.smoother.gradient(u, mu, tol)

    def hessvec_smooth(self, u, v, mu, tol=1e-12):
        return self.smoother.hessvec(u, v, mu, tol)

    def relative_obstacle_violation(self, u):
        with torch.no_grad():
            y = self.solve_state(u)
            violation = torch.relu(self.psi - y)
            return torch.norm(violation) / max(torch.norm(self.psi), torch.tensor(1e-14, device=u.device))

    def plot_arrays(self, u):
        with torch.no_grad():
            y = self.solve_state(u)
        return y, self.yd, self.psi