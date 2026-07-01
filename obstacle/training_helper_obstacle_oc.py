import torch
import numpy as np
import matplotlib.pyplot as plt

from semismooth_TR.tr_smoothing import trustregion

from .obstacle_oc_obj import ObstacleControlObjective, BoxL1Norm
from .smoothing_assistant import SmoothingAssistant


class L2Vector:
    def norm(self, x):
        return torch.norm(x)

    def dual(self, x):
        return x


class _ProblemWrap:
    pass


def train_obstacle_oc_with_TR(
    n=64,
    alpha=1e-3,
    beta=1e-5,
    gamma=1e2,
    delta0=1.0,
    maxit=300,
    device="cpu",
    mu_smooth=1e-4,
    deltamin=1e-4,
    deltamax=10.0,
):
    obj_smooth = ObstacleControlObjective(
        n=n,
        alpha=alpha,
        gamma=gamma,
        device=device,
    )

    obj_nonsmooth = BoxL1Norm(
        beta=beta,
        umin=-5.0,
        umax=5.0,
    )

    def f_smooth(u, mu):
        return obj_smooth.value_smooth_torch(u, mu)

    obj_smooth.attach_smoother(
        SmoothingAssistant(f_smooth)
    )

    problem = _ProblemWrap()
    problem.obj_smooth = obj_smooth
    problem.obj_nonsmooth = obj_nonsmooth
    problem.pvector = L2Vector()
    problem.dvector = L2Vector()

    u0 = torch.zeros(
        (n * n, 1),
        device=device,
        dtype=torch.get_default_dtype(),
    )

    params = {
        "spsolver": "SPG2",
        "delta": delta0,
        "deltamin": deltamin,
        "deltamax": deltamax,
        "maxit": maxit,
        "eta1": 0.1,
        "eta2": 0.75,
        "gamma1": 0.5,
        "gamma2": 2.0,
        "gtol": 1e-6,
        "stol": 1e-10,
        "ocScale": 1.0,
        "atol": 1e-6,
        "rtol": 1e-3,
        "spexp": 2,
        "outFreq": 1,
        "initProx": False,
        "use_smoothing_at_deltamin": True,
        "mu_smooth": mu_smooth,
        "boundary_tol": 0.8,
        "pred_abs_tol": 1e-12,
        "pred_rel_tol": 1e-12,
        "pred_small_max": 5,
        "nonmono_M": 10,
        "delta_stop": deltamin,
        "stol_abs": 1e-12,
        "stag_window": 20,
        "ftol_rel": 1e-8,
        "max_reject": 20,
    }

    u_opt, cnt, best_u = trustregion(u0, delta0, problem, params)

    return obj_smooth, u_opt, cnt, best_u


def plot_obstacle_oc(obj, u):
    y, yd, psi = obj.plot_arrays(u)

    n = obj.n
    X = obj.xy[:, 0].detach().cpu().numpy().reshape(n, n)
    Y = obj.xy[:, 1].detach().cpu().numpy().reshape(n, n)

    y = y.detach().cpu().numpy().reshape(n, n)
    yd = yd.detach().cpu().numpy().reshape(n, n)
    psi = psi.detach().cpu().numpy().reshape(n, n)

    violation = np.maximum(psi - y, 0.0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)

    im0 = axes[0].pcolormesh(X, Y, y, shading="auto")
    axes[0].set_title("state y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, yd, shading="auto")
    axes[1].set_title("desired y_d")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, Y, psi, shading="auto")
    axes[2].set_title("obstacle psi")
    fig.colorbar(im2, ax=axes[2])

    im3 = axes[3].pcolormesh(X, Y, violation, shading="auto")
    axes[3].set_title("max(psi - y, 0)")
    fig.colorbar(im3, ax=axes[3])

    plt.show()


def plot_tr_history(cnt):
    obj = np.array(cnt.get("objhist", []), dtype=float)
    gnm = np.array(cnt.get("gnormhist", []), dtype=float)
    delt = np.array(cnt.get("deltahist", []), dtype=float)
    snm = np.array(cnt.get("snormhist", []), dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)

    axes[0].plot(obj)
    axes[0].set_title("Objective")
    axes[0].set_xlabel("iter")

    axes[1].semilogy(gnm)
    axes[1].set_title("prox-gradient norm")
    axes[1].set_xlabel("iter")

    axes[2].semilogy(delt)
    axes[2].set_title("Delta")
    axes[2].set_xlabel("iter")

    axes[3].semilogy(snm)
    axes[3].set_title("step norm")
    axes[3].set_xlabel("iter")

    plt.show()