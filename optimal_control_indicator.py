import torch
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import math
from collections import deque

# ============================================================
# 0) Global settings
# ============================================================

torch.set_default_dtype(torch.float64)


# ============================================================
# 1) Optimization variable: discrete control vector
# ============================================================

class ControlVector:
    """
    Optimization variable = control values on interior grid.
    Stored as tensor of shape (m,1).
    """
    def __init__(self, data):
        self.data = data.clone()

    def copy(self):
        return ControlVector(self.data.detach().clone())

    def clone(self):
        return ControlVector(self.data.clone())

    def zero_like(self):
        return ControlVector(torch.zeros_like(self.data))

    def randn_like(self):
        return ControlVector(torch.randn_like(self.data))

    def __add__(self, other):
        return ControlVector(self.data + other.data)

    def __sub__(self, other):
        return ControlVector(self.data - other.data)

    def __mul__(self, a: float):
        return ControlVector(a * self.data)

    def __rmul__(self, a: float):
        return self.__mul__(a)

    def __imul__(self, a: float):
        self.data = a * self.data
        return self

    def __iadd__(self, other):
        self.data = self.data + other.data
        return self

    def __isub__(self, other):
        self.data = self.data - other.data
        return self

    def axpy(self, a: float, x: "ControlVector"):
        self.data = self.data + a * x.data
        return self

    def scal(self, a: float):
        self.data = a * self.data
        return self

    def dot(self, other) -> float:
        return float(torch.sum(self.data * other.data).item())

    def norm(self) -> float:
        return float(torch.sqrt(torch.sum(self.data * self.data)).item())

    def normalize_(self, eps: float = 1e-16):
        n = self.norm()
        if n > eps:
            self.data /= n
        return self


# ============================================================
# 2) Parameters
# ============================================================

def set_default_parameters(name):
    params = {}

    params['spsolver']  = name.replace(' ', '')
    params['outFreq']   = 1
    params['debug']     = False
    params['initProx']  = False
    params['t']         = 1.0
    params['safeguard'] = np.sqrt(np.finfo(float).eps)

    params['maxit']   = 200
    params['reltol']  = False
    params['gtol']    = 1e-7
    params['stol']    = 1e-12
    params['ocScale'] = params['t']

    params['eta1']     = 1e-4
    params['eta2']     = 0.75
    params['gamma1']   = 0.25
    params['gamma2']   = 2.5
    params['delta']    = 1.0
    params['deltamin'] = 1e-8
    params['deltamax'] = 100.0

    params['atol']    = 1e-5
    params['rtol']    = 1e-3
    params['spexp']   = 2
    params['maxitsp'] = 50

    params['useGCP']    = False
    params['mu1']       = 1e-4
    params['beta_dec']  = 0.1
    params['beta_inc']  = 10.0
    params['maxit_inc'] = 2

    params['lam_min'] = 1e-12
    params['lam_max'] = 1e12

    params["nonmono_M"] = 10
    return params


# ============================================================
# 3) Vector spaces for TR
# ============================================================

class EuclideanPrimal:
    def __init__(self, var=None):
        self.var = var

    @torch.no_grad()
    def dot(self, x, y):
        return float(torch.sum(x.data * y.data).item())

    @torch.no_grad()
    def norm(self, x):
        return float(torch.sqrt(torch.sum(x.data * x.data)).item())


class EuclideanDual:
    def __init__(self, var=None):
        self.var = var

    @torch.no_grad()
    def apply(self, x, y):
        return float(torch.sum(x.data * y.data).item())

    @torch.no_grad()
    def dual(self, x):
        return x


class L2TVPrimal(EuclideanPrimal):
    pass


class L2TVDual(EuclideanDual):
    pass


# ============================================================
# 4) Sampling / grid
# ============================================================

def make_training_points_grid(n=32, device="cpu", dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    xs = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return X, Y, xy


def interior_mask_2d(n, device="cpu"):
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    mask[1:-1, 1:-1] = True
    return mask.reshape(-1)


def build_fd_laplacian_2d(n, h, device="cpu", dtype=torch.float64):
    """
    Matrix for -Delta on the interior (n-2)x(n-2) grid.
    """
    ni = n - 2
    m = ni * ni
    A = torch.zeros((m, m), dtype=dtype, device=device)

    def idx(i, j):
        return i * ni + j

    c = 1.0 / (h * h)
    for i in range(ni):
        for j in range(ni):
            k = idx(i, j)
            A[k, k] = 4.0 * c
            if i > 0:
                A[k, idx(i - 1, j)] = -c
            if i < ni - 1:
                A[k, idx(i + 1, j)] = -c
            if j > 0:
                A[k, idx(i, j - 1)] = -c
            if j < ni - 1:
                A[k, idx(i, j + 1)] = -c
    return A


# ============================================================
# 5) Exact example from the paper
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
# 6) Semismooth nonlinearity max(0,y)
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


# ============================================================
# 8) Nonsmooth term phi = I_{C_ad}
# ============================================================

class IndicatorBox:
    """
    phi(u) = I_{C_ad}(u), C_ad = {u_a <= u <= u_b}
    """
    def __init__(self, var):
        self.var = var

    @torch.no_grad()
    def value(self, x):
        u_a = float(self.var["u_a"])
        u_b = float(self.var["u_b"])
        ok = torch.all(x.data >= u_a) and torch.all(x.data <= u_b)
        return 0.0 if ok else np.inf

    @torch.no_grad()
    def prox(self, x, t):
        u_a = float(self.var["u_a"])
        u_b = float(self.var["u_b"])
        return ControlVector(torch.clamp(x.data, min=u_a, max=u_b))

    def get_parameter(self):
        return float(self.var["u_a"]), float(self.var["u_b"])


# ============================================================
# 9) Smooth reduced objective
# ============================================================

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


# ============================================================
# 10) Problem wrapper
# ============================================================

class Problem:
    def __init__(self, obj_smooth, obj_nonsmooth, var=None):
        self.var = {} if var is None else dict(var)
        self.obj_smooth = obj_smooth
        self.obj_nonsmooth = obj_nonsmooth
        self.pvector = L2TVPrimal(self.var)
        self.dvector = L2TVDual(self.var)


# ============================================================
# 11) Trust-region pieces
# ============================================================

def dbls(nval, y, s, tmax, lambda_, kappa, gs, maxit, problem, cnt):
    tol = math.sqrt(np.finfo(float).eps)
    tol0 = 1e4 * tol
    eps0 = 1e2 * np.finfo(float).eps
    eps1 = 1e-2 * tol
    lam = 0.5 * (3 - math.sqrt(5))
    mu = 1e-4
    nold = nval

    tL = 0.0
    pL = 0.0
    tR = tmax

    pwa = y + tR * s
    nR = problem.obj_nonsmooth.value(pwa)
    cnt["nobj2"] += 1

    if np.isinf(nR):
        t0 = lambda_
        pwa = y + t0 * s
        n0 = problem.obj_nonsmooth.value(pwa)
        cnt["nobj2"] += 1
        p0 = t0 * (0.5 * t0 * kappa + gs) + n0 - nold

        if p0 >= 0:
            tR = t0
            pR = p0
            nR = n0
        else:
            t2 = tR
            tolBS = tol * (tR - lambda_)
            while t2 > t0 + tolBS:
                t1 = 0.5 * (t0 + t2)
                pwa = y + t1 * s
                n1 = problem.obj_nonsmooth.value(pwa)
                cnt["nobj2"] += 1
                if np.isinf(n1):
                    t2 = t1
                else:
                    tp = t0
                    pp = p0
                    t0 = t1
                    n0 = n1
                    p0 = t0 * (0.5 * t0 * kappa + gs) + n0 - nold
                    if p0 >= pp:
                        tL = tp
                        pL = pp
                        break
            tR = t0
            pR = p0
            nR = n0
    else:
        pR = tR * (0.5 * tR * kappa + gs) + nR - nold

    t = tR
    if kappa > 0:
        t = min(tR, -(((nR - nold) / max(tR - tL, 1e-30)) + gs) / kappa)

    useOpT = True
    if t <= tL:
        t = tL + lam * (tR - tL)
        useOpT = False

    pwa = y + t * s
    nt = problem.obj_nonsmooth.value(pwa)
    cnt["nobj2"] += 1

    Qt = t * gs + nt - nold
    pt = 0.5 * kappa * t**2 + Qt

    if useOpT and nt == nold and nR == nold:
        return t, nold, cnt

    if pt >= max(pL, pR):
        return tR, nR, cnt

    v = tL
    pv = pL
    w = tR
    pw = pR
    d = 0.0
    e = max(t - tL, tR - t)
    tm = 0.5 * (tL + tR)

    for _ in range(maxit):
        dL = tL - t
        dR = tR - t
        tol1 = tol0 * abs(t) + eps1
        tol2 = 2 * tol1
        tol3 = eps0 * max(abs(Qt), 1.0)

        if abs(e) > tol1:
            r = (t - w) * (pt - pv)
            q = (t - v) * (pt - pw)
            p = (t - v) * q - (t - w) * r
            q = 2 * (q - r)

            if q > 0:
                p = -p
            q = abs(q)

            etmp = e
            e = d

            if abs(p) >= abs(0.5 * q * etmp) or p <= q * dL or p >= q * dR:
                e = dR if t <= tm else dL
                d = lam * e
            else:
                d = p / q
                u = t + d
                if (u - tL < tol2) or (tR - u < tol2):
                    d = tol1 if tm >= t else -tol1
        else:
            e = dR if t <= tm else dL
            d = lam * e

        u = t + d
        if abs(d) < tol1:
            u = t + tol1 if d >= 0 else t - tol1

        pwa = y + u * s
        nu = problem.obj_nonsmooth.value(pwa)
        cnt["nobj2"] += 1

        Qu = u * gs + nu - nold
        pu = 0.5 * kappa * u**2 + Qu

        if pu <= pt:
            if u >= t:
                tL = t
            else:
                tR = t
            v, pv = w, pw
            w, pw = t, pt
            t, pt = u, pu
            nt, Qt = nu, Qu
        else:
            if u < t:
                tL = u
            else:
                tR = u
            if pu <= pw or w == t:
                v, pv = w, pw
                w, pw = u, pu
            elif pu <= pv or v == t or v == w:
                v, pv = u, pu

        tm = 0.5 * (tL + tR)

        if pt <= (mu * min(0.0, Qt) + tol3) and abs(t - tm) <= (tol2 - 0.5 * (tR - tL)):
            break

    return t, nt, cnt


def trustregion_step_NCG(x, val, dgrad, phi, problem, params, cnt):
    params.setdefault("debug", False)
    params.setdefault("gradTol", np.sqrt(np.finfo(float).eps))
    params.setdefault("t", 1.0)
    params.setdefault("safeguard", np.sqrt(np.finfo(float).eps))
    params.setdefault("maxitsp", 15)
    params.setdefault("atol", 1e-4)
    params.setdefault("rtol", 1e-2)
    params.setdefault("spexp", 2)
    params.setdefault("maxitdbls", 5)
    params.setdefault("ncg_type", 4)
    params.setdefault("eta", 1e-2)
    params.setdefault("descPar", 0.2)
    params.setdefault("lam_min", 1e-12)
    params.setdefault("lam_max", 1e12)

    del2 = params["delta"] ** 2
    snorm0 = 0.0
    ss0 = 0.0
    y = copy.deepcopy(x)
    gmod = copy.deepcopy(dgrad)
    valold = val
    phiold = phi

    Hg, _ = problem.obj_smooth.hessVec(dgrad, x, params["gradTol"])
    cnt["nhess"] = cnt.get("nhess", 0) + 1

    gHg = problem.dvector.apply(Hg, dgrad)
    gg = problem.pvector.dot(dgrad, dgrad)

    if gHg > params["safeguard"] * gg:
        lambdaTmp = gg / gHg
    else:
        lambdaTmp = params["t"] / np.sqrt(max(gg, 1e-30))

    lambda_ = np.max([params["lam_min"], np.min([params["lam_max"], lambdaTmp])])
    sc = problem.obj_nonsmooth.prox(y - lambda_ * gmod, lambda_) - y
    cnt["nprox"] = cnt.get("nprox", 0) + 1

    snormc = problem.pvector.norm(sc)
    lam1 = lambda_
    dx = (1.0 / lambda_) * sc
    s = copy.deepcopy(dx)
    gs = problem.pvector.dot(gmod, s)
    snorm = snormc / lambda_
    gnorm = snorm
    gtol = np.min([params["rtol"] * gnorm ** params["spexp"], params["atol"]])

    iflag = 1
    iter_count = 0
    phiold = phi

    for iter0 in range(1, params["maxitsp"] + 1):
        Hs, _ = problem.obj_smooth.hessVec(s, x, params["gradTol"])
        cnt["nhess"] = cnt.get("nhess", 0) + 1
        sHs = problem.dvector.apply(Hs, s)

        ds = problem.pvector.dot(s, y - x)
        ss = snorm ** 2
        alphaMax = (-ds + np.sqrt(max(ds**2 + ss * (del2 - ss0), 0.0))) / max(ss, 1e-30)

        alpha, phiold, cnt = dbls(
            phiold, y, s, alphaMax, lam1, sHs, gs,
            params["maxitdbls"], problem, cnt
        )

        y = y + alpha * s
        gmod = gmod + alpha * problem.dvector.dual(Hs)
        valold = valold + alpha * (gs + 0.5 * alpha * sHs)

        ss0 = alpha**2 * ss + 2 * alpha * ds + ss0
        snorm0 = np.sqrt(max(ss0, 0.0))

        if snorm0 >= (1 - params["safeguard"]) * params["delta"]:
            iflag = 2
            iter_count = iter0
            break

        if sHs > params["safeguard"] * ss:
            lambdaTmp = ss / sHs
        else:
            lambdaTmp = params["t"] / max(problem.pvector.norm(gmod), 1e-30)

        lambda_ = np.max([params["lam_min"], np.min([params["lam_max"], lambdaTmp])])

        dx0 = copy.deepcopy(dx)
        gnorm0 = gnorm
        dx = (1.0 / lambda_) * (problem.obj_nonsmooth.prox(y - lambda_ * gmod, lambda_) - y)
        cnt["nprox"] = cnt.get("nprox", 0) + 1

        gnorm = problem.pvector.norm(dx)

        if gnorm <= gtol:
            iflag = 0
            iter_count = iter0
            break

        gnorm2 = gnorm ** 2
        d = dx0 - dx

        if params["ncg_type"] == 4:
            denom = problem.pvector.dot(s, d)
            beta = max(0.0, gnorm2 / max(denom, 1e-30))
        else:
            beta = 0.0

        reset = True
        if beta != 0 and np.isfinite(beta):
            s0 = dx + beta * s
            gs0 = problem.pvector.dot(gmod, s0)
            phi_trial = problem.obj_nonsmooth.value(y + s0)
            cnt["nobj2"] = cnt.get("nobj2", 0) + 1

            if (gs0 + phi_trial - phiold) <= -(1 - params["descPar"]) * gnorm2:
                s = s0
                gs = gs0
                lam1 = 1.0
                reset = False

        if reset:
            s = copy.deepcopy(dx)
            lam1 = lambda_
            gs = problem.pvector.dot(gmod, s)

        snorm = problem.pvector.norm(s)
        iter_count = iter0

    s = y - x
    snorm = problem.pvector.norm(s)
    phinew = phiold
    pRed = (val + phi) - (valold + phinew)

    return s, snorm, pRed, phinew, iflag, iter_count, cnt, params


def compute_gradient(x, problem, params, cnt):
    gtol = 1e-12
    grad, gerr = problem.obj_smooth.gradient(x, gtol)
    cnt['ngrad'] += 1
    dgrad = problem.dvector.dual(grad)
    pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
    cnt['nprox'] += 1
    gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']

    params['gradTol'] = gtol
    cnt.setdefault('graderr', []).append(gerr)
    cnt.setdefault('gradtol', []).append(gtol)
    return grad, dgrad, gnorm, cnt


def trustregion(x0, Deltai, problem, params):
    start_time = time.time()

    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 500)
    params.setdefault('gtol', 1e-7)
    params.setdefault('stol', 1e-9)
    params.setdefault('ocScale', 1.0)
    params.setdefault('atol', 1e-4)
    params.setdefault('rtol', 1e-2)
    params.setdefault('spexp', 2)
    params.setdefault('eta1', 1e-4)
    params.setdefault('eta2', 0.5)
    params.setdefault('gamma1', 0.25)
    params.setdefault('gamma2', 1.5)
    params.setdefault('delta', Deltai)
    params.setdefault('deltamin', 1e-16)
    params.setdefault('deltamax', 100.0)
    params.setdefault('reltol', False)
    params.setdefault('delta_stop', 1e-7)
    params.setdefault('stol_abs', 1e-9)
    params.setdefault('stag_window', 10)
    params.setdefault('ftol_rel', 1e-6)
    params.setdefault('max_reject', 15)
    params.setdefault("nonmono_M", 10)
    params.setdefault("pred_abs_tol", 1e-11)
    params.setdefault("pred_rel_tol", 1e-11)
    params.setdefault("pred_small_max", 5)
    params.setdefault("useInexactGrad", False)

    cnt = {
        'AlgType': f"TR-{params.get('spsolver','NCG')}",
        'iter': 0,
        'nobj1': 0,
        'ngrad': 0,
        'nobj2': 0,
        'nprox': 0,
        'nhess': 0,
        'timetotal': 0.0,
        'objhist': [],
        'obj1hist': [],
        'obj2hist': [],
        'gnormhist': [],
        'snormhist': [],
        'deltahist': [],
        'nobj1hist': [],
        'nobj2hist': [],
        'ngradhist': [],
        'nproxhist': [],
        'timehist': [],
        'graderr': [],
        'gradtol': []
    }

    obj = problem.obj_smooth

    if params['initProx']:
        x = problem.obj_nonsmooth.prox(x0, 1.0)
        cnt['nprox'] += 1
    else:
        x = x0.copy()

    obj.update(x, "init")

    rej_count = 0
    small_pred_count = 0

    val_true, _ = obj.value(x, 1e-12)
    cnt['nobj1'] += 1

    val_model = val_true
    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)

    phi = problem.obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    Facc = [val_true + phi]
    Fhist = deque(maxlen=params["nonmono_M"])
    Fhist.append(val_true + phi)

    print(f"TR method using {params.get('spsolver','NCG')} Subproblem Solver")
    print("  iter    value    gnorm    del    snorm    nobjs    ngrad    nobjn    nprox     iterSP    flagSP")
    print(f"{0:4d} {0:4d} {val_true+phi:8.6e} {params['delta']:8.6e}  ---      "
          f"{cnt['nobj1']:6d}      {cnt['ngrad']:6d}      {cnt['nobj2']:6d}      {cnt['nprox']:6d} --- ---")

    cnt['objhist'].append(val_true + phi)
    cnt['obj1hist'].append(val_true)
    cnt['obj2hist'].append(phi)
    cnt['gnormhist'].append(gnorm)
    cnt['snormhist'].append(np.nan)
    cnt['deltahist'].append(params['delta'])
    cnt['nobj1hist'].append(cnt['nobj1'])
    cnt['nobj2hist'].append(cnt['nobj2'])
    cnt['ngradhist'].append(cnt['ngrad'])
    cnt['nproxhist'].append(cnt['nprox'])
    cnt['timehist'].append(np.nan)

    gtol = params['gtol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm

    for i in range(1, params['maxit'] + 1):
        params['tolsp'] = min(params['atol'], params['rtol'] * (gnorm ** params['spexp']))
        grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)

        val_model = val_true
        cnt['nobj1'] += 1

        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_NCG(
            x, val_model, dgrad, phi, problem, params, cnt
        )

        pRed = float(pRed)
        pred_floor = max(
            params["pred_abs_tol"],
            params["pred_rel_tol"] * max(1.0, abs(val_model + phi))
        )

        if pRed <= pred_floor:
            small_pred_count += 1
        else:
            small_pred_count = 0

        if (small_pred_count >= params["pred_small_max"]) and (params["delta"] <= 10.0 * params["delta_stop"]):
            cnt['iter'] = i
            cnt['timetotal'] = time.time() - start_time
            cnt['iflag'] = 6
            print("Optimization terminated because predicted reduction is tiny repeatedly.")
            return x, cnt

        xnew = x + s

        valnew_true, _ = obj.value(xnew, 1e-12)
        cnt['nobj1'] += 1
        phinew_true = problem.obj_nonsmooth.value(xnew)
        cnt['nobj2'] += 1

        aRed = (val_true + phi) - (valnew_true + phinew_true)

        rho = -np.inf if pRed <= 0.0 else float(aRed) / pRed
        Fref = max(Fhist)
        accept_nm = (valnew_true + phinew_true) <= (Fref - 1e-12)
        accept = (rho >= params['eta1']) and accept_nm

        if not accept:
            params['delta'] = max(params['deltamin'], params['gamma1'] * params['delta'])
            obj.update(x, 'reject')
            rej_count += 1
        else:
            x = xnew
            phi = phinew_true
            val_true = valnew_true
            rej_count = 0

            obj.update(x, 'accept')
            Facc.append(val_true + phi)
            Fhist.append(val_true + phi)

            relL2u = obj.relative_L2_error_control(x)
            relL2y = obj.relative_L2_error_state(x)
            print(f"   relL2(control)={relL2u:.3e}, relL2(state)={relL2y:.3e}")

            if rho > params['eta2']:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])

        if i % params['outFreq'] == 0:
            print(f"{i:4d}    {val_true + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      "
                  f"{cnt['nobj1']:6d}     {cnt['ngrad']:6d}       {cnt['nobj2']:6d}     {cnt['nprox']:6d}      "
                  f"{iter_count:4d}        {iflag:1d}")

        cnt['objhist'].append(val_true + phi)
        cnt['obj1hist'].append(val_true)
        cnt['obj2hist'].append(phi)
        cnt['gnormhist'].append(gnorm)
        cnt['snormhist'].append(snorm)
        cnt['deltahist'].append(params['delta'])
        cnt['nobj1hist'].append(cnt['nobj1'])
        cnt['nobj2hist'].append(cnt['nobj2'])
        cnt['ngradhist'].append(cnt['ngrad'])
        cnt['nproxhist'].append(cnt['nprox'])
        cnt['timehist'].append(time.time() - start_time)

        delta_stop = params["delta_stop"]
        stol_abs   = params["stol_abs"]
        K          = params["stag_window"]
        ftol_rel   = params["ftol_rel"]
        max_reject = params["max_reject"]

        stop_grad = (gnorm <= gtol)
        stop_step = (snorm < stol_abs) and (params["delta"] <= delta_stop)
        stop_stag = False
        if len(Facc) >= K + 1:
            Fold = Facc[-(K+1)]
            Fnew = Facc[-1]
            rel_change = abs(Fold - Fnew) / max(1.0, abs(Fnew))
            stop_stag = (rel_change < ftol_rel)
        stop_stuck = (params["delta"] <= 10 * delta_stop and rej_count >= max_reject)
        stop_maxit = (i >= params["maxit"])

        if stop_grad or stop_step or stop_stag or stop_stuck or stop_maxit:
            if stop_grad:
                flag = 0
                reason = "gradient tolerance met"
            elif stop_step:
                flag = 2
                reason = "step small and TR radius collapsed"
            elif stop_stag:
                flag = 3
                reason = "objective stagnation"
            elif stop_stuck:
                flag = 4
                reason = "trust region stuck"
            else:
                flag = 1
                reason = "maximum iterations reached"

            cnt['iter'] = i
            cnt['timetotal'] = time.time() - start_time
            cnt['iflag'] = flag

            print("Optimization terminated because", reason)
            print(f"Total time: {cnt['timetotal']:8.6e} seconds")
            return x, cnt

    cnt['iter'] = params['maxit']
    cnt['timetotal'] = time.time() - start_time
    cnt['iflag'] = 1
    return x, cnt


# ============================================================
# 12) Derivative check / Hessian check
# ============================================================

def directional_fd_value(obj, theta, v, eps):
    th_p = theta.copy()
    th_m = theta.copy()
    th_p.axpy(eps, v)
    th_m.axpy(-eps, v)
    Jp, _ = obj.value(th_p)
    Jm, _ = obj.value(th_m)
    return (Jp - Jm) / (2.0 * eps)


def directional_grad_dot(obj, theta, v):
    g, _ = obj.gradient(theta)
    return g.dot(v)


def grad_check(obj, theta, ntests=3, eps_list=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    for t in range(ntests):
        v = theta.randn_like()
        v.normalize_()
        gTv = directional_grad_dot(obj, theta, v)
        print(f"\nTest {t}: g^T v = {gTv:+.6e}")
        for eps in eps_list:
            fd = directional_fd_value(obj, theta, v, eps)
            relerr = abs(fd - gTv) / max(1.0, abs(fd), abs(gTv))
            print(f" eps={eps: >8.1e} FD={fd:+.6e} relerr={relerr:.3e}")


def directional_fd_grad(obj, theta, v, eps):
    th_p = theta.copy()
    th_m = theta.copy()
    th_p.axpy(eps, v)
    th_m.axpy(-eps, v)
    gp, _ = obj.gradient(th_p)
    gm, _ = obj.gradient(th_m)
    out = gp.copy()
    out.axpy(-1.0, gm)
    out.scal(1.0 / (2.0 * eps))
    return out


def hv_check(obj, theta, ntests=3, eps_list=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    for t in range(ntests):
        v = theta.randn_like()
        v.normalize_()
        Hv, _ = obj.hessVec(v, theta)
        print(f"\nTest {t}: ||Hv|| = {Hv.norm():.6e}")
        for eps in eps_list:
            fdHv = directional_fd_grad(obj, theta, v, eps)
            diff = fdHv.copy()
            diff.axpy(-1.0, Hv)
            relerr = diff.norm() / max(1.0, fdHv.norm(), Hv.norm())
            print(f" eps={eps:>8.1e} ||FD-Hv||/scale={relerr:.3e}")


# ============================================================
# 13) Driver
# ============================================================

def solve_optimal_control_with_TR(
    ngrid=64,
    alpha=1e-2,
    u_a=-1e6,
    u_b=1e6,
    delta0=1.0,
    maxit=50,
    device="cpu",
):
    X, Y, xy = make_training_points_grid(ngrid, device=device)
    g_d = desired_state(xy)
    y_true = y_star(xy)
    f_rhs = f_source(xy, alpha)

    var = {
        "useEuclidean": True,
        "u_a": u_a,
        "u_b": u_b,
    }

    obj_smooth = ReducedSemilinearControlObjective(
        xy=xy,
        g_d=g_d,
        y_true = y_true,
        f_rhs=f_rhs,
        alpha=alpha,
        ngrid=ngrid,
        weight=None,
        device=device,
        mu_I=0.0,
        newton_tol=1e-10,
        newton_maxit=30,
        fd_eps=1e-6,
    )

    obj_nonsmooth = IndicatorBox(var)
    problem = Problem(obj_smooth, obj_nonsmooth, var)

    m = (ngrid - 2) * (ngrid - 2)
    x0 = ControlVector(torch.zeros((m, 1), dtype=torch.get_default_dtype(), device=device))

    params = set_default_parameters("NCG")
    params["delta"] = delta0
    params["maxit"] = maxit
    params["gtol"] = 1e-7
    params["useInexactObj"] = False
    params["useInexactGrad"] = False

    x_opt, cnt = trustregion(x0, delta0, problem, params)
    return x_opt, cnt, problem, X, Y, xy, g_d, f_rhs


# ============================================================
# 14) Helpers for plotting final state/control
# ============================================================

def embed_interior_to_full(u_int, ngrid, device="cpu", dtype=torch.float64):
    mask = interior_mask_2d(ngrid, device=device)
    full = torch.zeros((ngrid * ngrid, 1), dtype=dtype, device=device)
    full[mask] = u_int
    return full


def compute_state_from_control(problem, x_opt):
    y_int = problem.obj_smooth.solve_state_cached(x_opt)
    y_full = embed_interior_to_full(
        y_int, problem.obj_smooth.ngrid,
        device=problem.obj_smooth.device,
        dtype=problem.obj_smooth.xy.dtype
    )
    u_full = embed_interior_to_full(
        x_opt.data, problem.obj_smooth.ngrid,
        device=problem.obj_smooth.device,
        dtype=problem.obj_smooth.xy.dtype
    )
    return y_full, u_full


# ============================================================
# 15) Plotting
# ============================================================

@torch.no_grad()
def plot_solution_and_error(problem, x_opt, X, Y, xy, g_d, alpha, device="cpu"):
    n = problem.obj_smooth.ngrid
    y_full, u_full = compute_state_from_control(problem, x_opt)

    y = y_full.reshape(n, n).detach().cpu().numpy()
    u = u_full.reshape(n, n).detach().cpu().numpy()
    gd = g_d.reshape(n, n).detach().cpu().numpy()

    uex = u_star(xy, alpha).reshape(n, n).detach().cpu().numpy()
    yerr = y - gd
    uerr = u - uex

    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    im0 = axes[0, 0].pcolormesh(Xn, Yn, gd, shading="auto")
    axes[0, 0].set_title("desired state $y^*$")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(Xn, Yn, y, shading="auto")
    axes[0, 1].set_title("computed state $y$")
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].pcolormesh(Xn, Yn, yerr, shading="auto")
    axes[0, 2].set_title("$y-y^*$")
    fig.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].pcolormesh(Xn, Yn, uex, shading="auto")
    axes[1, 0].set_title("exact control $u^*$")
    fig.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(Xn, Yn, u, shading="auto")
    axes[1, 1].set_title("computed control $u$")
    fig.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].pcolormesh(Xn, Yn, uerr, shading="auto")
    axes[1, 2].set_title("$u-u^*$")
    fig.colorbar(im5, ax=axes[1, 2])

    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.show()


def plot_tr_history(cnt):
    obj = np.array(cnt.get("objhist", []), dtype=float)
    gnm = np.array(cnt.get("gnormhist", []), dtype=float)
    delt = np.array(cnt.get("deltahist", []), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    if len(obj) > 0:
        axes[0].plot(obj)
        axes[0].set_title("Objective")
        axes[0].set_xlabel("iter")

    if len(gnm) > 0:
        axes[1].plot(gnm)
        axes[1].set_title("prox-gradient norm")
        axes[1].set_xlabel("iter")
        axes[1].set_yscale("log")

    if len(delt) > 0:
        axes[2].plot(delt)
        axes[2].set_title("TR radius")
        axes[2].set_xlabel("iter")
        axes[2].set_yscale("log")

    plt.show()


# ============================================================
# 16) Main
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ngrid = 64
    alpha = 1e-2


    u_a = -0.5
    u_b =  0.5

    X, Y, xy = make_training_points_grid(ngrid, device=device)
    g_d = desired_state(xy)
    y_true = y_star(xy)
    f_rhs = f_source(xy, alpha)

    obj_smooth = ReducedSemilinearControlObjective(
        xy=xy,
        g_d=g_d,
        y_true=y_true,
        f_rhs=f_rhs,
        alpha=alpha,
        ngrid=ngrid,
        device=device,
        mu_I=0.0,
        newton_tol=1e-10,
        newton_maxit=30,
        fd_eps=1e-6,
    )

    m = (ngrid - 2) * (ngrid - 2)
    x0 = ControlVector(torch.zeros((m, 1), dtype=torch.get_default_dtype(), device=device))

    print("\n==== GRAD CHECK at x0 ====")
    grad_check(obj_smooth, x0, ntests=3)

    print("\n==== HESSIAN CHECK at x0 ====")
    hv_check(obj_smooth, x0, ntests=3)

    x_opt, cnt, problem, X, Y, xy, g_d, f_rhs = solve_optimal_control_with_TR(
        ngrid=ngrid,
        alpha=alpha,
        u_a=u_a,
        u_b=u_b,
        delta0=1.0,
        maxit=50,
        device=device,
    )

    print("\nFinal objective:", cnt["objhist"][-1])
    print("Termination flag:", cnt["iflag"])

    rel_u = problem.obj_smooth.relative_L2_error_control(x_opt)
    rel_y = problem.obj_smooth.relative_L2_error_state(x_opt)
    print("Relative L2 error in control:", rel_u)
    print("Relative L2 error in state:  ", rel_y)

    plot_solution_and_error(problem, x_opt, X, Y, xy, g_d, alpha, device=device)
    plot_tr_history(cnt)
     
