import torch
from collections import OrderedDict
from .data_generation import make_shepp_logan_data
from .Saturation import SaturationLeastSquares
from .waveletl1_tv import TVNonsmooth , L1Nonsmooth, WaveletL1DetailNonsmooth, WaveletL1Nonsmooth
from .Problem import Problem
from semismooth_TR.TorchVector import TorchDictVector
import matplotlib.pyplot as plt
import numpy as np

def make_initial_guess(n, device="cpu", mode="data", b=None):
    if mode == "zeros":
        img = torch.zeros((n, n), dtype=torch.float64, device=device)
    elif mode == "data":
        if b is None:
            raise ValueError("b must be provided when mode='data'")
        img = b.detach().clone()
    else:
        raise ValueError("mode must be 'zeros' or 'data'")

    return TorchDictVector(OrderedDict(img=img))


def build_tv_saturation_problem(
    n=128,
    sigma=2.0,
    kernel_size=9,
    noise_level=0.01,
    lam=1e-3,
    device="cpu",
    x0_mode="data"
):
    x_true, b, A_apply, AT_apply, _ = make_shepp_logan_data(
        n=n,
        sigma=sigma,
        kernel_size=kernel_size,
        noise_level=noise_level,
        device=device,
    )

    obj_smooth = SaturationLeastSquares(
        b=b,
        A_apply=A_apply,
        AT_apply=AT_apply,
        x_true=x_true,
        key="img",
        kink_tol=1e-8,
    )

    obj_nonsmooth = TVNonsmooth(
        lam=lam,
        key="img",
        prox_max_iter=2000,
        prox_tol=1e-10,
    )

    problem = Problem(
        obj_smooth=obj_smooth,
        obj_nonsmooth=obj_nonsmooth,
        var={"useEuclidean": True},
    )

    x0 = make_initial_guess(n=n, device=device, mode=x0_mode, b=b)
    return x_true, b, x0, problem

def build_l1_saturation_problem(
    n=128,
    sigma=2.0,
    kernel_size=9,
    noise_level=0.01,
    lam=1e-3,
    device="cpu",
    x0_mode="data"
):
    x_true, b, A_apply, AT_apply, _ = make_shepp_logan_data(
        n=n,
        sigma=sigma,
        kernel_size=kernel_size,
        noise_level=noise_level,
        device=device,
    )

    obj_smooth = SaturationLeastSquares(
        b=b,
        A_apply=A_apply,
        AT_apply=AT_apply,
        x_true=x_true,
        key="img",
        kink_tol=1e-8,
    )

    obj_nonsmooth = L1Nonsmooth(
        lam=lam,
        key="img",
    )

    problem = Problem(
        obj_smooth=obj_smooth,
        obj_nonsmooth=obj_nonsmooth,
        var={"useEuclidean": True},
    )

    x0 = make_initial_guess(n=n, device=device, mode=x0_mode, b=b)
    return x_true, b, x0, problem

def build_wavelet_l1_saturation_problem(
    n=128,
    sigma=2.0,
    kernel_size=9,
    noise_level=0.01,
    lam=2e-4,
    device="cpu",
    x0_mode="zeros",
    wavelet="db2",
    level=2,
):
    x_true, b, A_apply, AT_apply, _ = make_shepp_logan_data(
        n=n,
        sigma=sigma,
        kernel_size=kernel_size,
        noise_level=noise_level,
        device=device,
    )

    obj_smooth = SaturationLeastSquares(
        b=b,
        A_apply=A_apply,
        AT_apply=AT_apply,
        x_true=x_true,
        key="img",
        kink_tol=1e-8,
    )

    obj_nonsmooth = WaveletL1DetailNonsmooth(
        lam=lam,
        key="img",
        wavelet=wavelet,
        level=level,
        mode="periodization",
    )

    problem = Problem(
        obj_smooth=obj_smooth,
        obj_nonsmooth=obj_nonsmooth,
        var={"useEuclidean": True},
    )

    x0 = make_initial_guess(n=n, device=device, mode=x0_mode, b=b)
    return x_true, b, x0, problem

def show_results(x_true, b, x_opt):
    xt = x_true.detach().cpu().numpy()
    bb = b.detach().cpu().numpy()
    rec = x_opt.td["img"].detach().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(xt, cmap="gray")
    plt.title("True image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(bb, cmap="gray")
    plt.title("Observed data")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(rec, cmap="gray")
    plt.title("Reconstruction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

@torch.no_grad()
def check_smooth_derivatives(problem,x, key="img",seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    obj=problem.obj_smooth
    s = x.zero_like()
    s.td[key]=torch.randn_like(x.td[key])
    s_norm = problem.pvector.norm(s)
    if s_norm == 0:
        raise RuntimeError("Random direction has zero norm")
    s = (1.0/s_norm)*s
    
    f0, _ = obj.value(x,1e-12)
    g0, _ = obj.gradient(x,1e-12)
    Hs, _ = obj.hessVec(s,x,1e-12)
    g0s = problem.pvector.dot(g0,s)
    print("n=== First-order derivative check ===")
    print("eps    ||g(x+eps s)-g(x)-eps Hs|| ratio/eps^2")
    for eps in [1e-1, 5e-2,1e-2,5e-3,1e-3,5e-4,1e-4]:
        xeps = x+eps*s
        geps,_ = obj.gradient(xeps,1e-12)
        rem = geps-g0-eps*Hs
        err2 = problem.pvector.norm(rem)
        print(f"{eps:8.1e}   {err2:26.6e} {err2/(eps**2):14.6e}")
        
@torch.no_grad()
def check_formula_interior(problem, x, key="img"):
    obj = problem.obj_smooth
    u = x.td[key]
    Ax = obj.A_apply(u)

    print("\n=== Interior formula check ===")
    print("min(Ax) =", torch.min(Ax).item(), " max(Ax) =", torch.max(Ax).item())

    g, _ = obj.gradient(x, 1e-12)
    g_exact = obj.AT_apply(Ax - obj.b)
    errg = torch.norm(g.td[key] - g_exact).item()
    print("||g - A^T(Ax-b)|| =", errg)

    s = x.zero_like()
    s.td[key] = torch.randn_like(u)
    Hs, _ = obj.hessVec(s, x, 1e-12)
    Hs_exact = obj.AT_apply(obj.A_apply(s.td[key]))
    errH = torch.norm(Hs.td[key] - Hs_exact).item()
    print("||Hs - A^T A s|| =", errH)
    
    
    
        
@torch.no_grad()
def make_interior_test_point(n,device="cpu"):
    img = 0.5+0.05*torch.randn((n,n),dtype=torch.float64,device=device)
    img = torch.clamp(img,0.2,0.8)
    return TorchDictVector(OrderedDict(img=img))

@torch.no_grad()
def check_clipping_margin(problem, x, key="img"):
    obj = problem.obj_smooth
    u = x.td[key]
    Ax = obj.A_apply(u)

    min_to_0 = torch.min(Ax).item()
    min_to_1 = torch.min(1.0 - Ax).item()

    print("\n=== Clipping margin diagnostic ===")
    print("min(Ax)      =", min_to_0)
    print("min(1 - Ax)  =", min_to_1)
    print("If both are comfortably positive, you're away from kinks.")
from matplotlib.ticker import MaxNLocator    
def plot_tr_performance(cnt, use_log=True):
    obj = np.asarray(cnt["objhist"], dtype=float)
    gnorm = np.asarray(cnt["gnormhist"], dtype=float)
    delta = np.asarray(cnt["deltahist"], dtype=float)

    it = np.arange(len(obj), dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

    axes[0].plot(it, obj,  markersize=3)
    
    axes[0].set_title("Objective")
    axes[0].set_xlabel("iter")
    axes[0].grid(False)

    axes[1].plot(it, gnorm,  markersize=3)
    
    axes[1].set_title("prox-gradient norm")
    axes[1].set_xlabel("iter")
    axes[1].grid(False)

    axes[2].plot(it, delta, markersize=3)
    axes[2].set_title("TR radius")
    axes[2].set_xlabel("iter")
    axes[2].grid(False)
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if use_log:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")

    plt.tight_layout()
    plt.show()
    
