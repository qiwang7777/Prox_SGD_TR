import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from semismooth_TR.set_default_parameters import set_default_parameters
from semismooth_TR.trust_region import trustregion

from .sampling import make_training_points_grid, interior_mask_2d
from .pde_solver import desired_state, y_star, f_source, u_star
from .Reduced_obj import ReducedSemilinearControlObjective
from .Indicator import IndicatorBox
from .Problem_wrapper import Problem
from .ControlVector import ControlVector


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


def solve_once(
    spsolver,
    ngrid=64,
    alpha=1e-2,
    u_a=-0.5,
    u_b=0.5,
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
        y_true=y_true,
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
    x0 = ControlVector(
        torch.zeros((m, 1), dtype=torch.get_default_dtype(), device=device)
    )

    params = set_default_parameters(spsolver)
    params["spsolver"] = spsolver
    params["delta"] = delta0
    params["maxit"] = maxit
    params["gtol"] = 1e-7
    params["useInexactObj"] = False
    params["useInexactGrad"] = False
    params["debug"] = False

    if spsolver.upper() == "SSN":
        params["ssn_maxit"] = 1
        params["ssn_reg"] = 1e-6
        params["ssn_cg_tol"] = 1e-6
        params["ssn_cg_maxit"] = 40
        params["ssn_tol"] = 1e-10

    t0 = time.time()
    x_opt, cnt, best_x = trustregion(x0, delta0, problem, params)
    elapsed = time.time() - t0

    rel_u = problem.obj_smooth.relative_L2_error_control(x_opt)
    rel_y = problem.obj_smooth.relative_L2_error_state(x_opt)

    return {
        "solver": spsolver,
        "x_opt": x_opt,
        "best_x": best_x,
        "cnt": cnt,
        "problem": problem,
        "X": X,
        "Y": Y,
        "xy": xy,
        "g_d": g_d,
        "f_rhs": f_rhs,
        "rel_u": rel_u,
        "rel_y": rel_y,
        "time": elapsed,
    }


def compare_solvers(
    solvers=("NCG", "SPG2", "SSN"),
    ngrid=64,
    alpha=1e-2,
    u_a=-0.5,
    u_b=0.5,
    delta0=1.0,
    maxit=50,
    device="cpu",
):
    results = {}
    rows = []

    for solver in solvers:
        print("\n" + "=" * 70)
        print(f"Running TR subsolver: {solver}")
        print("=" * 70)

        res = solve_once(
            spsolver=solver,
            ngrid=ngrid,
            alpha=alpha,
            u_a=u_a,
            u_b=u_b,
            delta0=delta0,
            maxit=maxit,
            device=device,
        )

        cnt = res["cnt"]
        iters = max(1,cnt.get("iter",1))

        row = {
            "solver": solver,
            "final_obj": cnt["objhist"][-1],
            "final_gnorm": cnt["gnormhist"][-1],
            "rel_u": res["rel_u"],
            "rel_y": res["rel_y"],
            "iters": iters,
            "iflag": cnt.get("iflag", None),
            "nobj1": cnt.get("nobj1", None),
            "nobj2": cnt.get("nobj2", None),
            "ngrad": cnt.get("ngrad", None),
            "nprox": cnt.get("nprox", None),
            "nhess": cnt.get("nhess", 0),
            "time_sec": res["time"],
            "time_per_iter": res["time"]/iters,
            "hess_per_iter": cnt.get("nhess",0)/iters
        }

        rows.append(row)
        results[solver] = res

    table = pd.DataFrame(rows)
    return table, results


def plot_histories(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    for solver, res in results.items():
        cnt = res["cnt"]

        obj = np.array(cnt.get("objhist", []), dtype=float)
        gnm = np.array(cnt.get("gnormhist", []), dtype=float)
        delt = np.array(cnt.get("deltahist", []), dtype=float)

        if len(obj) > 0:
            axes[0].plot(obj, label=solver)

        if len(gnm) > 0:
            axes[1].plot(gnm, label=solver)

        if len(delt) > 0:
            axes[2].plot(delt, label=solver)

    axes[0].set_title("Objective")
    axes[0].set_xlabel("TR iteration")

    axes[1].set_title("Prox-gradient norm")
    axes[1].set_xlabel("TR iteration")
    axes[1].set_yscale("log")

    axes[2].set_title("TR radius")
    axes[2].set_xlabel("TR iteration")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.show()

def plot_timing_comparison(table):
    fig, ax = plt.subplots(figsize=(6, 4))

    solvers = table["solver"]
    times = table["time_sec"]

    ax.bar(solvers, times)

    ax.set_ylabel("time (seconds)")
    ax.set_title("Wall-clock runtime comparison")

    for i, t in enumerate(times):
        ax.text(i, t, f"{t:.2f}s", ha="center", va="bottom")

    ax.grid(True, axis="y", alpha=0.3)

    plt.show()

def save_comparison_table(table, filename="tr_subsolver_comparison.csv"):
    table.to_csv(filename, index=False)
    print(f"Saved comparison table to {filename}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    table, results = compare_solvers(
        solvers=("NCG", "SPG2", "SSN"),
        ngrid=64,
        alpha=1e-2,
        u_a=-0.5,
        u_b=0.5,
        delta0=1.0,
        maxit=50,
        device=device,
    )

    print("\n==== TR subsolver comparison ====")
    print(table.to_string(index=False))

    save_comparison_table(table)
    plot_histories(results)
    plot_timing_comparison(table)
