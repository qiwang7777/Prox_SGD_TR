import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from semismooth_TR.TorchVector import TorchDictVector
from .Poisson_obj import u_star, PoissonCompositeObjective
from .qReLU_NN import PoissonNet
from .L1torch import L1TorchNorm
from img_l1.DictEuclidean import L2TVDual, L2TVPrimal
from semismooth_TR.set_default_parameters import set_default_parameters
from semismooth_TR.trust_region import trustregion

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
})

def make_training_points_grid(n=32, device="cpu", dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    xs = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return xy

def kappa_xy(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    return torch.ones_like(x)



def compute_g_from_u_star(xy: torch.Tensor) -> torch.Tensor:
    """
    g(x) = -div( kappa(x) * grad u*(x) ), computed by AD.
    """
    xy_req = xy.detach().clone().requires_grad_(True)

    u = u_star(xy_req)  # (N,1)
    grad_u = torch.autograd.grad(u.sum(), xy_req, create_graph=True)[0]  # (N,2)

    kap = kappa_xy(xy_req)  # (N,1)
    flux = kap * grad_u     # (N,2)

    div_flux = 0.0
    for j in range(2):
        div_flux = div_flux + torch.autograd.grad(
            flux[:, j].sum(), xy_req, create_graph=True
        )[0][:, j:j+1]  # (N,1)

    g = -div_flux.detach()
    return g

def vector_from_model(model: torch.nn.Module) -> TorchDictVector:
    td = OrderedDict()
    for name, p in model.named_parameters():
        td[name] = p.detach().clone()
    return TorchDictVector(td)


def load_vector_into_model(x: TorchDictVector, model: torch.nn.Module):
    with torch.no_grad():
        name_to_param = dict(model.named_parameters())
        for k, v in x.td.items():
            name_to_param[k].copy_(v)


def train_poisson_with_TR(
    width=64, depth=2, ngrid=32,
    beta=0.0,
    delta0=1.0,
    maxit=1,
    device="cpu",
):
   
    model = PoissonNet(width=width, depth=depth, out_dim=1, activation='quadratic').to(device)
    
    # data
    xy = make_training_points_grid(ngrid, device=device)
    g = compute_g_from_u_star(xy)  # manufactured forcing on same grid
    

    # objective + problem wrapper
    var = {
        "useEuclidean": False,
        "beta": beta
    }

    
    h_w = 1.0/(ngrid - 1)
    weight = torch.tensor(h_w * h_w, dtype = xy.dtype, device = device)
    obj_smooth = PoissonCompositeObjective(
        model=model,
        xy=xy,
        g=g,
        kappa_fn=kappa_xy,
        weight=weight,
        device=device,
        mu_I=0.0,
        xb=None,
        wb=None,
        bc_target = None,
        lam_bc=0.0,
        x_true = True,
        u_true_fn = u_star,
    )
    obj_smooth.set_hess_mode("full")
    obj_nonsmooth = L1TorchNorm(var)

    class _ProblemWrap:
        pass
    problem = _ProblemWrap()
    problem.obj_smooth = obj_smooth
    problem.obj_nonsmooth = obj_nonsmooth
    problem.pvector = L2TVPrimal(var)
    problem.dvector = L2TVDual(var)

    # initial parameter vector
    x0 = vector_from_model(model)

    # TR params
    params = set_default_parameters("SPG2")
    params["delta"] = delta0
    params["maxit"] = maxit
    params["useInexactObj"] = False
    params["useInexactGrad"] = False

    # if you implemented these stopping rules in TR:
    params["pred_floor_rel"] = 1e-12
    params["ared_accept_rel"] = 1e-12
    params["rej_max_stuck_at_floor"] = 10

    # run TR
    x_opt, cnt, best_x = trustregion(x0, delta0, problem, params)

    # load optimum back into model
    load_vector_into_model(x_opt, model)
    return model, x_opt, cnt, best_x




# -------------------------
# 5) Plotting (consistent with PoissonNet output)
# -------------------------

@torch.no_grad()
def plot_solution_and_error(model, n=101, device="cpu"):
    xs = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.get_default_dtype())
    ys = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.get_default_dtype())
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    # PoissonNet already outputs u(x)
    u_pred = model(xy).reshape(n, n).detach().cpu().numpy()
    u_ref  = u_star(xy).reshape(n, n).detach().cpu().numpy()
    err    = u_pred - u_ref

    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    im0 = axes[0].pcolormesh(Xn, Yn, u_pred, shading="auto")
    axes[0].set_title("u_pred")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(Xn, Yn, u_ref, shading="auto")
    axes[1].set_title("u_star (truth)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(Xn, Yn, err, shading="auto")
    axes[2].set_title("u_pred - u_star")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    plt.show()

def plot_tr_history(cnt):
    obj = np.array(cnt.get("objhist", []), dtype=float)
    gnm = np.array(cnt.get("gnormhist", []), dtype=float)
    delt = np.array(cnt.get("deltahist", []), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    if len(obj) > 0:
        axes[0].plot(obj)
        axes[0].set_title("Objective (val + phi)")
        axes[0].set_xlabel("iter")

    if len(gnm) > 0:
        axes[1].plot(gnm)
        axes[1].set_title("gnorm (prox-grad norm)")
        axes[1].set_xlabel("iter")
        axes[1].set_yscale("log")

    if len(delt) > 0:
        axes[2].plot(delt)
        axes[2].set_title("Trust-region radius delta")
        axes[2].set_xlabel("iter")
        axes[2].set_yscale("log")

    plt.show()




