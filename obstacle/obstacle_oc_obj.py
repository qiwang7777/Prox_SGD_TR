import torch
from collections import OrderedDict
from semismooth_TR.TorchVector import TorchDictVector


def make_grid(n=64, device="cpu", dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()

    xs = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)

    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    h = 1.0 / (n - 1)

    return xy, h


def make_control_vector(n, device="cpu", dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()

    td = OrderedDict()
    td["u"] = torch.zeros((n * n, 1), device=device, dtype=dtype)

    return TorchDictVector(td)


def get_u(x):
    if isinstance(x, TorchDictVector):
        return x.td["u"]
    return x


def psi_fun(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    return 0.08 * torch.exp(-40.0 * ((x - 0.5)**2 + (y - 0.5)**2))



def yd_fun(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    return -0.12 * torch.exp(-30.0 * ((x - 0.5)**2 + (y - 0.5)**2))



def softplus_mu(z, mu):
    return mu * torch.nn.functional.softplus(z / mu)


class BoxL1Norm:
    """
    phi(u) = beta ||u||_1 + indicator_{[umin, umax]}(u).

    Works with TorchDictVector x where x.td["u"] is the control.
    """

    def __init__(self, beta=1e-5, umin=-5.0, umax=5.0):
        self.beta = beta
        self.umin = umin
        self.umax = umax

    def value(self, x):
        u = get_u(x)

        in_box = torch.all((u >= self.umin) & (u <= self.umax))
        if not bool(in_box.detach().cpu().item()):
            return float("inf")

        return self.beta * torch.sum(torch.abs(u))

    def prox(self, x, r):
        u = get_u(x)

        z = torch.sign(u) * torch.clamp(
            torch.abs(u) - r * self.beta,
            min=0.0,
        )

        z = torch.clamp(z, self.umin, self.umax)

        if isinstance(x, TorchDictVector):
            td = OrderedDict()
            td["u"] = z
            return TorchDictVector(td)

        return z


class ObstacleControlObjective:
    """
    Obstacle-control objective:

        f(u) =
            0.5 ||y - y_d||^2
          + 0.5 alpha ||u||^2
          + 0.5 gamma ||max(0, psi - y)||^2

    subject to

        -Delta y = u, y = 0 on boundary.

    The nonsmooth part handled by prox is:

        phi(u) = beta ||u||_1 + box constraint.
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

        self.x_true = None

    def _idx(self, i, j):
        return i * self.n + j

    def _build_poisson_matrix(self):
        n = self.n
        h = self.h
        N = n * n

        A = torch.zeros((N, N), dtype=self.dtype, device=self.device)

        for i in range(n):
            for j in range(n):
                p = self._idx(i, j)

                if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                    A[p, p] = 1.0
                else:
                    A[p, p] = 4.0 / h ** 2
                    A[p, self._idx(i - 1, j)] = -1.0 / h ** 2
                    A[p, self._idx(i + 1, j)] = -1.0 / h ** 2
                    A[p, self._idx(i, j - 1)] = -1.0 / h ** 2
                    A[p, self._idx(i, j + 1)] = -1.0 / h ** 2

        return A

    def solve_state_tensor(self, u):
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

    def solve_state(self, x):
        u = get_u(x)
        return self.solve_state_tensor(u)

    def value_torch(self, x):
        u = get_u(x)
        y = self.solve_state_tensor(u)

        tracking = 0.5 * self.weight * torch.sum((y - self.yd) ** 2)
        control = 0.5 * self.alpha * self.weight * torch.sum(u ** 2)

        violation = self.psi - y

        obstacle = 0.5 * self.gamma * self.weight * torch.sum(
            torch.relu(violation) ** 2
        )

        return tracking + control + obstacle

    def value_smooth_torch(self, x, mu):
        u = get_u(x)
        y = self.solve_state_tensor(u)

        tracking = 0.5 * self.weight * torch.sum((y - self.yd) ** 2)
        control = 0.5 * self.alpha * self.weight * torch.sum(u ** 2)

        violation = self.psi - y
        smooth_violation = softplus_mu(violation, mu)

        obstacle = 0.5 * self.gamma * self.weight * torch.sum(
            smooth_violation ** 2
        )

        return tracking + control + obstacle

    def value(self, x, tol=1e-12):
        val = self.value_torch(x)
        return float(val.detach().cpu().item()), 0.0

    def gradient(self, x, tol=1e-12):
        u = get_u(x)

        u_ad = u.detach().clone().requires_grad_(True)

        if isinstance(x, TorchDictVector):
            td = OrderedDict()
            td["u"] = u_ad
            x_ad = TorchDictVector(td)
        else:
            x_ad = u_ad

        val = self.value_torch(x_ad)

        grad_u = torch.autograd.grad(
            val,
            u_ad,
            create_graph=False,
            retain_graph=False,
        )[0]

        if isinstance(x, TorchDictVector):
            tdg = OrderedDict()
            tdg["u"] = grad_u.detach()
            return TorchDictVector(tdg), 0.0

        return grad_u.detach(), 0.0

    def hessVec(self, v, x, tol=1e-12):
        u = get_u(x)
        vu = get_u(v)

        u_ad = u.detach().clone().requires_grad_(True)

        if isinstance(x, TorchDictVector):
            td = OrderedDict()
            td["u"] = u_ad
            x_ad = TorchDictVector(td)
        else:
            x_ad = u_ad

        val = self.value_torch(x_ad)

        grad_u = torch.autograd.grad(
            val,
            u_ad,
            create_graph=True,
            retain_graph=True,
        )[0]

        gv = torch.sum(grad_u * vu)

        hv_u = torch.autograd.grad(
            gv,
            u_ad,
            retain_graph=False,
        )[0]

        if isinstance(x, TorchDictVector):
            tdh = OrderedDict()
            tdh["u"] = hv_u.detach()
            return TorchDictVector(tdh), 0.0

        return hv_u.detach(), 0.0

    def attach_smoother(self, smoother):
        self.smoother = smoother

    def update(self, x, flag=None):
        pass

    def value_smooth(self, x, mu, tol=1e-12):
        if not hasattr(self, "smoother"):
            raise RuntimeError("No smoother has been attached.")
        return self.smoother.value(x, mu, tol)

    def gradient_smooth(self, x, mu, tol=1e-12):
        if not hasattr(self, "smoother"):
            raise RuntimeError("No smoother has been attached.")
        return self.smoother.gradient(x, mu, tol)

    def hessvec_smooth(self, v, x, mu, tol=1e-12):
        if not hasattr(self, "smoother"):
            raise RuntimeError("No smoother has been attached.")
        return self.smoother.hessVec(v, x, mu, tol)

    def relative_obstacle_violation(self, x):
        with torch.no_grad():
            y = self.solve_state(x)
            violation = torch.relu(self.psi - y)

            denom = torch.norm(self.psi)
            denom = torch.clamp(denom, min=torch.tensor(1e-14, device=self.device, dtype=self.dtype))

            return torch.norm(violation) / denom

    def relative_L2_error(self, x):
        return self.relative_obstacle_violation(x)

    def plot_arrays(self, x):
        with torch.no_grad():
            y = self.solve_state(x)

        return y, self.yd, self.psi
