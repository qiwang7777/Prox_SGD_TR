import numpy as np
import torch
import matplotlib.pyplot as plt

from semismooth_TR.set_default_parameters import set_default_parameters
from semismooth_TR.trust_region import trustregion
from semismooth_TR.derivative_check import grad_check

from .GridVector import GridVector
from .L1 import L1Penalty
from .Problem_wrapper import Problem
from .ReLU_operator_obj import SparseReLUOperatorObjective
from .sampling import make_grid, target_function


torch.set_default_dtype(torch.float64)

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
    }
)


def solve_sparse_relu_operator_regression(
    ngrid=48,
    alpha=1e-4,
    beta=1e-7,
    sigma=0.08,
    delta0=1.0,
    maxit=100,
    device="cpu",
    relu_zero_selection="midpoint",
):
    X, Y, xy = make_grid(ngrid, device=device)
    g = target_function(xy)

    h = 1.0 / (ngrid - 1)
    var = {
        "useEuclidean": True,
        "beta": beta,
        "weight": h * h,
    }

    obj_smooth = SparseReLUOperatorObjective(
        xy=xy,
        g=g,
        alpha=alpha,
        sigma=sigma,
        ngrid=ngrid,
        weight=h * h,
        device=device,
        relu_zero_selection=relu_zero_selection,
        zero_tol=1e-12,
        fd_eps=1e-6,
    )
    obj_nonsmooth = L1Penalty(var)
    problem = Problem(obj_smooth, obj_nonsmooth, var)

    x0 = GridVector(
        torch.zeros(
            (ngrid * ngrid, 1),
            dtype=torch.get_default_dtype(),
            device=device,
        )
    )

    # SPG2 uses only first-order/proximal information and is the natural
    # default for this semismooth composite example.
    params = set_default_parameters("SPG2")
    params["delta"] = delta0
    params["maxit"] = maxit
    params["gtol"] = 1e-7
    params["useInexactObj"] = False
    params["useInexactGrad"] = False

    x_opt, cnt, best_x = trustregion(x0, delta0, problem, params)
    return x_opt, cnt, problem, X, Y, xy, g


@torch.no_grad()
def compute_fields(problem, x):
    n = problem.obj_smooth.ngrid
    u = x.data.reshape(n, n)
    Ku = problem.obj_smooth.apply_K(x).reshape(n, n)
    prediction = torch.relu(Ku)
    return u, Ku, prediction


@torch.no_grad()
def plot_solution(problem, x_opt, X, Y, g):
    n = problem.obj_smooth.ngrid
    u, Ku, prediction = compute_fields(problem, x_opt)

    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()
    un = u.detach().cpu().numpy()
    Kun = Ku.detach().cpu().numpy()
    predn = prediction.detach().cpu().numpy()
    gn = g.reshape(n, n).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)

    im0 = axes[0].pcolormesh(Xn, Yn, gn, shading="auto")
    axes[0].set_title("target $g$")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[1].pcolormesh(Xn, Yn, predn, shading="auto")
    axes[1].set_title(r"$\operatorname{ReLU}(Ku)$")
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[2].pcolormesh(Xn, Yn, un, shading="auto")
    axes[2].set_title("recovered sparse control $u$")
    fig.colorbar(im2, ax=axes[1, 0])

    im3 = axes[3].pcolormesh(Xn, Yn, Kun, shading="auto")
    axes[3].contour(Xn, Yn, Kun, levels=[0.0], linewidths=1.5)
    axes[3].set_title("$Ku$ with ReLU interface")
    fig.colorbar(im3, ax=axes[1, 1])

    for ax in axes.ravel():
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    plt.show()


def plot_tr_history(cnt):
    obj = np.asarray(cnt.get("objhist", []), dtype=float)
    gnm = np.asarray(cnt.get("gnormhist", []), dtype=float)
    delt = np.asarray(cnt.get("deltahist", []), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    if obj.size:
        axes[0].plot(obj)
        axes[0].set_title("Objective")
        axes[0].set_xlabel("iteration")

    if gnm.size:
        axes[1].plot(gnm)
        axes[1].set_title("prox-gradient norm")
        axes[1].set_xlabel("iteration")
        axes[1].set_yscale("log")

    if delt.size:
        axes[2].plot(delt)
        axes[2].set_title("trust-region radius")
        axes[2].set_xlabel("iteration")
        axes[2].set_yscale("log")

    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ngrid = 48
    alpha = 1e-4
    beta = 1e-7
    sigma = 0.08

    X, Y, xy = make_grid(ngrid, device=device)
    g = target_function(xy)

    # Optional generalized-gradient sanity check at a nonzero test point.
    h = 1.0 / (ngrid - 1)
    obj_test = SparseReLUOperatorObjective(
        xy=xy,
        g=g,
        alpha=alpha,
        sigma=sigma,
        ngrid=ngrid,
        weight=h * h,
        device=device,
        relu_zero_selection="midpoint",
    )
    x_test = GridVector(
        1e-2
        * torch.randn(
            (ngrid * ngrid, 1),
            dtype=torch.get_default_dtype(),
            device=device,
        )
    )
    print("\n==== GENERALIZED GRADIENT CHECK ====")
    grad_check(obj_test, x_test, ntests=3)

    x_opt, cnt, problem, X, Y, xy, g = solve_sparse_relu_operator_regression(
        ngrid=ngrid,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        delta0=1.0,
        maxit=100,
        device=device,
        relu_zero_selection="midpoint",
    )

    print("\nFinal objective:", cnt["objhist"][-1])
    print("Termination flag:", cnt["iflag"])
    print("ReLU active fraction:", problem.obj_smooth.relu_active_fraction(x_opt))
    print("ReLU kink fraction:", problem.obj_smooth.relu_kink_fraction(x_opt))
    print(
        "Fraction of near-zero coefficients:",
        float(torch.mean((torch.abs(x_opt.data) <= 1e-8).to(x_opt.data.dtype)).item()),
    )

    plot_solution(problem, x_opt, X, Y, g)
    plot_tr_history(cnt)
