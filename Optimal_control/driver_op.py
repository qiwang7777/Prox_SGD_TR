import torch
from semismooth_TR.set_default_parameters import set_default_parameters
from semismooth_TR.trust_region import trustregion
from semismooth_TR.derivative_check import grad_check, hv_check
from .sampling import make_training_points_grid,interior_mask_2d
from .pde_solver import desired_state, y_star, f_source, u_star
from .Reduced_obj import ReducedSemilinearControlObjective
from .Indicator import IndicatorBox
from .Problem_wrapper import Problem
from .ControlVector import ControlVector
import matplotlib.pyplot as plt
import numpy as np

torch.set_default_dtype(torch.float64)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
})

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
    yd = g_d.reshape(n, n).detach().cpu().numpy()
    yex = y_star(xy).reshape(n,n).detach().cpu().numpy()

    uex = u_star(xy, alpha).reshape(n, n).detach().cpu().numpy()
    yerr = y - yex
    uerr = u - uex

    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    im0 = axes[0, 0].pcolormesh(Xn, Yn, yex, shading="auto")
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