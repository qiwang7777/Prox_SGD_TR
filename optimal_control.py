# We take the same example as in " A descent algorithm for the optimal control of ReLU neural network informed PDEs based on approximate directional derivatives"
#min J(y,u) = 0.5*||y-g||^2+0.5*alpha*||u||^2. s.t. -∆y+N( ,y)=u in Ω,and y=0 on ∂Ω

import torch
import numpy as np
import copy,time
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
import torch.nn.functional as F
from torch.func import functional_call
# -------------------------------
# 0) Nonlinearity and exact data
# -------------------------------

class MonotoneReLUlinearity(nn.Module):
    """
    Scalar monotone ReLU nonlinearity N(y).
    N(y) = c0*y+sum_i a_i*ReLU(b_i*y+c_i)
    To keep N increasing in y, choose: c0>=0, a_i>=0, b_i>=0
    """
    def __init__(self,c0=1.0, a=(1.0,0.5),b=(1.0,2.0),c=(0.0,-0.25)):
        super().__init__()
        self.c0 = float(c0)
        self.a = [float(v) for v in a]
        self.b = [float(v) for v in b]
        self.c = [float(v) for v in c]
        assert len(self.a) == len(self.b) == len(self.c)
        
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        out = self.c0 * y
        for ai, bi, ci in zip(self.a, self.b, self.c):
            out = out + ai * torch.relu(bi * y + ci)
        return out

def monotone_relu_derivative(nonlinearity: MonotoneReLUlinearity, y: torch.Tensor) -> torch.Tensor:
    """
    N'(y) for
      N(y) = c0*y + sum_i a_i * ReLU(b_i*y + c_i)
    """
    out = torch.full_like(y, nonlinearity.c0)
    for ai, bi, ci in zip(nonlinearity.a, nonlinearity.b, nonlinearity.c):
        out = out + ai * bi * (bi * y + ci > 0).to(y.dtype)
    return out

def y_star_oc(xy:torch.Tensor)->torch.Tensor:
    x = xy[:,0:1]
    y = xy[:,1:2]
    return torch.sin(math.pi*x)*torch.sin(math.pi*y)

def minus_laplace_y_star_oc(xy: torch.Tensor) -> torch.Tensor:
    yv = y_star_oc(xy)
    return 2.0 * (math.pi ** 2) * yv

def u_star_oc(xy: torch.Tensor, nonlinearity: nn.Module) -> torch.Tensor:
    yv = y_star_oc(xy)
    return minus_laplace_y_star_oc(xy) + nonlinearity(yv)

def g_target_from_optimality(
    xy: torch.Tensor,
    nonlinearity: MonotoneReLUlinearity,
    alpha: float
) -> torch.Tensor:
    """
    Build g so that (y*, u*, p*) satisfies the full optimality system:
      -Δy* + N(y*) = u*
      -Δp* + N'(y*) p* = y* - g
      alpha*u* + p* = 0
    """
    xy_req = xy.detach().clone().requires_grad_(True)

    yv = y_star_oc(xy_req)                                  # (N,1)
    Ny = nonlinearity(yv)                                   # (N,1)
    u_star = 2.0 * (math.pi ** 2) * yv + Ny                # -Δy* + N(y*)
    p_star = -alpha * u_star                                # stationarity

    # Compute Δp* by AD
    grad_p = torch.autograd.grad(p_star.sum(), xy_req, create_graph=True)[0]  # (N,2)

    p_x = grad_p[:, 0:1]
    p_y = grad_p[:, 1:2]

    p_xx = torch.autograd.grad(p_x.sum(), xy_req, create_graph=True)[0][:, 0:1]
    p_yy = torch.autograd.grad(p_y.sum(), xy_req, create_graph=True)[0][:, 1:2]

    lap_p = p_xx + p_yy
    Nprime_y = monotone_relu_derivative(nonlinearity, yv)

    g = yv + lap_p - Nprime_y * p_star
    return g.detach()

# -------------------------
# 1) Networks
# -------------------------

def b_factor(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return x*(1-x)*y*(1-y)

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, mapping_size=8, scale=1.0, trainable=False):
        super().__init__()
        B = torch.randn((in_dim, mapping_size)) * scale
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x):
        # x: (N, 2)
        x_proj = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class MLP(nn.Module):
    """
    Plain MLP: x -> h(f(x))
    Activation can be 'relu' or 'softplus' etc.
    """
    def __init__(self, in_dim=2, width=64, depth=3, out_dim=1, activation="relu", smooth_beta=50.0):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "softplus":
            act = nn.Softplus(beta=1.0)   # smooth ReLU-ish
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation = activation
        self.smooth_beta = smooth_beta

        layers = []
        layers += [nn.Linear(in_dim, width), act]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act]
        layers += [nn.Linear(width, out_dim)]

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # Kaiming init works well for ReLU/Softplus/GELU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in = m.weight.size(1)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x, smooth=False):
        if (not smooth) or (self.activation != "relu"):
            return self.net(x)
        h = x
        beta = self.smooth_beta
        for layer in self.net:
            if isinstance(layer,nn.ReLU):
                h = F.softplus(h,beta = beta)
            else:
                h = layer(h)
        
        return h
    
class MLPWithFourier(nn.Module):
    def __init__(self, in_dim=2, mapping_size=8, scale=1.0,
                 width=64, depth=3, out_dim=1, activation="relu", trainable_B=False, smooth_beta=50.0):
        super().__init__()
        self.ff = FourierFeatures(in_dim=in_dim, mapping_size=mapping_size, scale=scale, trainable=trainable_B)
        self.mlp = MLP(in_dim=2 * mapping_size, width=width, depth=depth, out_dim=out_dim,
                       activation=activation, smooth_beta=smooth_beta)

    def forward(self, x, smooth=False):
        x = self.ff(x)
        return self.mlp(x, smooth=smooth)
    
    
       
    
class StateNet(nn.Module):
    """
    Hard BC state network:
        y_theta(x)=b_factor(x)*raw_net(x)
    """
    def __init__(self,in_dim=2, width=64, depth=3, activation="relu"):
        super().__init__()
        self.raw = MLPWithFourier(in_dim = in_dim,
                                  mapping_size = 8,
                                  scale = 0.5,
                                  width = width,
                                  depth = depth,
                                  out_dim = 1,
                                  activation = activation)
        
    def forward(self, xy, smooth=False):
        return b_factor(xy)*self.raw(xy,smooth=smooth)
    
class ControlNet(nn.Module):
    """
    Control network u_theta(x)
    No Hard BC needed
    """
    
    def __init__(self, in_dim=2, width=64, depth=3, activation="relu"):
        super().__init__()
        self.net = MLPWithFourier(in_dim = in_dim,
                                  mapping_size = 8,
                                  scale = 0.5,
                                  width = width,
                                  depth = depth,
                                  out_dim = 1,
                                  activation = activation)
        
    def forward(self,xy,smooth=False):
        
        return self.net(xy,smooth=smooth)
        
class AdjointNet(nn.Module):
    def __init__(self, in_dim=2, width=64, depth=3, activation="relu"):
        super().__init__()
        self.raw = MLPWithFourier(
            in_dim=in_dim,
            mapping_size=8,
            scale=0.5,
            width=width,
            depth=depth,
            out_dim=1,
            activation=activation
        )

    def forward(self, xy, smooth=False):
        return b_factor(xy) * self.raw(xy,smooth=smooth)
    








# -------------------------------
# 2) TorchDictVector and helpers
# -------------------------------


class TorchDictVector:
    """
    A lightweight vector wrapper for NN parameters:
    - stores parameters in a dict: .td[name] = tensor
    - supports +, -, scalar *, deep-ish copy/clone
    """
    def __init__(self, td=None):
        self.td = OrderedDict() if td is None else OrderedDict(td)

    def copy(self):
        # copy with detached tensors (keeps values)
        return TorchDictVector({k: v.detach().clone() for k, v in self.td.items()})

    def clone(self):
        # clone keeping graph if present (rarely needed here)
        return TorchDictVector({k: v.clone() for k, v in self.td.items()})

    def zero_like(self):
        return TorchDictVector({k: torch.zeros_like(v) for k, v in self.td.items()})

    # --- algebra ---
    def __add__(self, other):
        out = TorchDictVector()
        for k in self.td.keys():
            out.td[k] = self.td[k] + other.td[k]
        return out

    def __sub__(self, other):
        out = TorchDictVector()
        for k in self.td.keys():
            out.td[k] = self.td[k] - other.td[k]
        return out

    def __mul__(self, a: float):
        out = TorchDictVector()
        for k, v in self.td.items():
            out.td[k] = a * v
        return out

    def __rmul__(self, a: float):
        return self.__mul__(a)

    def __iadd__(self, other):
        for k in self.td.keys():
            self.td[k] = self.td[k] + other.td[k]
        return self

    def __isub__(self, other):
        for k in self.td.keys():
            self.td[k] = self.td[k] - other.td[k]
        return self
    
    def randn_like(self):
        return TorchDictVector({k: torch.randn_like(v) for k, v in self.td.items()})
    
    def dot(self, other) -> float:
        s = 0.0
        for k, v in self.td.items():
            s += torch.sum(v * other.td[k]).item()
        return float(s)
    
    def axpy(self, a: float, x: "TorchDictVector"):
        for k in self.td.keys():
            self.td[k] = self.td[k] + a * x.td[k]
        return self
    
    def scal(self, a: float):
        for k in self.td.keys():
            self.td[k] = a * self.td[k]
        return self
    
    def norm(self) -> float:
        s = 0.0
        for v in self.td.values():
            s += torch.sum(v * v).item()
        return float(s ** 0.5)
    def normalize_(self, eps: float = 1e-16):
        n = self.norm()
        if n < eps:
            return self
        return self.scal(1.0 / n)



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


# =========================================================
# 3) Linear algebra wrappers and nonsmooth term
# =========================================================
class DictEuclidean:
    """
    Euclidean inner product on TorchDictVector:
      <x,y> = sum_k sum_i x_k[i]*y_k[i]
    """
    def __init__(self, var=None):
        self.var = var

    @torch.no_grad()
    def dot(self, x, y):
        s = 0.0
        for k, vx in x.td.items():
            s += torch.sum(vx * y.td[k]).item()
        return float(s)

    @torch.no_grad()
    def apply(self, x, y):
        
        return self.dot(x, y)

    @torch.no_grad()
    def norm(self, x):
        return float(np.sqrt(self.dot(x, x)))

    @torch.no_grad()
    def dual(self, x):
        # identity for Euclidean metric
        return x

class L2TVPrimal(DictEuclidean):
    pass

class L2TVDual(DictEuclidean):
    pass

class ControlOnlyL1TorchNorm:
    """
    phi(x) = beta * ||theta_u||_1
    Only control::* entries are penalized/prox'ed.
    state::* entries are untouched.
    """
    def __init__(self, var):
        self.var = var

    @torch.no_grad()
    def value(self, x):
        beta = float(self.var["beta"])
        s = 0.0
        for k, v in x.td.items():
            if k.startswith("control::"):
                s += torch.sum(torch.abs(v)).item()
        return beta * float(s)

    @torch.no_grad()
    def prox(self, x, t):
        beta = float(self.var["beta"])
        tau = t * beta
        out = x.copy()
        for k, v in x.td.items():
            if k.startswith("control::"):
                out.td[k] = torch.sign(v) * torch.clamp(torch.abs(v) - tau, min=0.0)
            else:
                out.td[k] = v.clone()
        return out

# =========================================================
# 4) Discretization helpers
# =========================================================

def make_training_points_grid(n=32, device="cpu", dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    xs = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return xy

def make_sine_tests(xy: torch.Tensor, modes=((1,1), (1,2), (2,1), (2,2))):
    """
    Returns:
      V   : (N, K)   values of test functions at xy
      dV  : (N, K, 2) gradients of test functions at xy
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    vals = []
    grads = []

    for m, n in modes:
        vx = torch.sin(math.pi * m * x)
        vy = torch.sin(math.pi * n * y)
        v = vx * vy

        dv_dx = math.pi * m * torch.cos(math.pi * m * x) * vy
        dv_dy = math.pi * n * vx * torch.cos(math.pi * n * y)
        dv = torch.stack([dv_dx.squeeze(-1), dv_dy.squeeze(-1)], dim=-1)  # (N,2)

        vals.append(v)      # (N,1)
        grads.append(dv)    # (N,2)

    V = torch.cat(vals, dim=1)              # (N,K)
    dV = torch.stack(grads, dim=1)          # (N,K,2)
    return V, dV

# =========================================================
# 5) Weak state and adjoint solvers
# =========================================================
class WeakSemilinearStateSolver: 
    """ 
    Solve the semilinear state equation in the weak form 
    ∫ grad(y)·grad(v_k) + ∫ N(y) v_k = ∫ u v_k 
    by minimizing the weak residual least-squares functional over the 
    parameters of a state network y_theta. 
    The state is represented by a hard-BC network, 
    so y=0 on the boundary. 
    """ 
    def __init__(self, 
                 state_model: nn.Module, 
                 nonlinearity: nn.Module, 
                 xy: torch.Tensor, 
                 V: torch.Tensor, 
                 dV: torch.Tensor, 
                 weight = None, 
                 device = "cpu", 
                 max_iter = 100, 
                 tol = 1e-8, 
                 lr = 1.0, 
                 verbose = False, 
                 warm_start = True,
                 ): 
        self.state_model = state_model.to(device) 
        self.nonlinearity = nonlinearity.to(device) 
        self.xy = xy.to(device) 
        self.V = V.to(device) 
        self.dV = dV.to(device) 
        self.device = device 
        self.max_iter = int(max_iter) 
        self.tol = float(tol) 
        self.lr = float(lr) 
        self.verbose = bool(verbose) 
        self.warm_start = bool(warm_start) 
        if weight is None: 
            self.weight = None 
        else: 
            self.weight =weight.to(device) if torch.is_tensor(weight) else weight 
        self._last_state = None 
        
    @torch.no_grad() 
    def reset_state(self): 
        self._last_state = None 
        
    @torch.no_grad() 
    def _save_state(self): 
        self._last_state = {k: v.detach().clone() for k,v in self.state_model.state_dict().items()} 
    
    @torch.no_grad() 
    def _load_state(self): 
        if self._last_state is not None: 
            self.state_model.load_state_dict(self._last_state) 
    
    def _weak_residual_vector(self, u_vals:torch.Tensor) -> torch.Tensor: 
        xy = self.xy.detach().clone().requires_grad_(True) 
        
        y = self.state_model(xy) 
        grad_y = torch.autograd.grad(y.sum(),xy, create_graph=True)[0] 
        Ny = self.nonlinearity(y) 
        
        gy_dot_dvk = (grad_y.unsqueeze(1)*self.dV).sum(dim=-1) 
        integrand = gy_dot_dvk + Ny * self.V - u_vals * self.V 
        
        if self.weight is None: 
            R = integrand.mean(dim=0) 
        else: 
            w = self.weight 
            if torch.is_tensor(w) and w.numel() == 1: 
                R = (w*integrand).sum(dim=0) 
            else: 
                if torch.is_tensor(w) and w.ndim == 1: 
                    w = w.reshape(-1, 1) 
                R = (w * integrand).sum(dim=0) 
        return R 
    
    def _residual_loss(self, u_vals:torch.Tensor) -> torch.Tensor: 
        R = self._weak_residual_vector(u_vals) 
        return 0.5 * torch.mean(R**2) 
    def solve(self, u_vals:torch.Tensor): 
        """ 
        Solve for the state y corresponding to fixed control values u_vals 
        on the training grid 
        Returns: 
            y_detached: (N,1) tensor 
            info: dict 
        """ 
        u_vals = u_vals.detach().to(self.device) 
        if self.warm_start: 
            self._load_state() 
            
        params = list(self.state_model.parameters()) 
        optimizer = torch.optim.LBFGS( 
            params, 
            lr = self.lr, 
            max_iter = self.max_iter, 
            tolerance_grad = self.tol, 
            tolerance_change = 1e-12, 
            history_size = 50, 
            line_search_fn="strong_wolfe")
        
        info= {"final_loss": None, "niter":0} 
    
        def closure(): 
            optimizer.zero_grad(set_to_none=True) 
            loss = self._residual_loss(u_vals) 
            loss.backward() 
            return loss 
    
        loss = optimizer.step(closure) 
        info["final_loss"] = float(loss.detach().cpu().item()) 
    
        with torch.no_grad(): 
             y = self.state_model(self.xy).detach() 
        
        self._save_state() 
        return y, info  

class WeakSemilinearAdjointSolver:
    def __init__(
        self,
        adjoint_model: nn.Module,
        nonlinearity: nn.Module,
        xy: torch.Tensor,
        V: torch.Tensor,
        dV: torch.Tensor,
        g_target: torch.Tensor,
        weight=None,
        device="cpu",
        max_iter=100,
        tol=1e-8,
        lr=1.0,
        verbose=False,
        warm_start=True,
    ):
        self.adjoint_model = adjoint_model.to(device)
        self.nonlinearity = nonlinearity.to(device)
        self.xy = xy.to(device)
        self.V = V.to(device)
        self.dV = dV.to(device)
        self.g_target = g_target.to(device)
        self.weight = None if weight is None else (weight.to(device) if torch.is_tensor(weight) else weight)
        self.device = device
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.lr = float(lr)
        self.verbose = bool(verbose)
        self.warm_start = bool(warm_start)
        self._last_state = None
        
    @torch.no_grad()
    def reset_state(self):
        self._last_state = None

    @torch.no_grad()
    def _save_state(self):
        self._last_state = {k: v.detach().clone() for k, v in self.adjoint_model.state_dict().items()}

    @torch.no_grad()
    def _load_state(self):
        if self._last_state is not None:
            self.adjoint_model.load_state_dict(self._last_state)

    def _weak_residual_vector(self, y_vals: torch.Tensor) -> torch.Tensor:
        xy = self.xy.detach().clone().requires_grad_(True)
        p = self.adjoint_model(xy)
        grad_p = torch.autograd.grad(p.sum(), xy, create_graph=True)[0]
        Nprime_y = monotone_relu_derivative(self.nonlinearity, y_vals.detach())

        gp_dot_dvk = (grad_p.unsqueeze(1) * self.dV).sum(dim=-1)
        integrand = gp_dot_dvk + (Nprime_y * p) * self.V - (y_vals.detach() - self.g_target) * self.V

        if self.weight is None:
            R = integrand.mean(dim=0)
        else:
            w = self.weight
            if torch.is_tensor(w) and w.numel() == 1:
                R = (w * integrand).sum(dim=0)
            else:
                if torch.is_tensor(w) and w.ndim == 1:
                    w = w.reshape(-1, 1)
                R = (w * integrand).sum(dim=0)
        return R

    def _residual_loss(self, y_vals: torch.Tensor) -> torch.Tensor:
        R = self._weak_residual_vector(y_vals)
        return 0.5 * torch.mean(R**2)

    def solve(self, y_vals: torch.Tensor):
        y_vals = y_vals.detach().to(self.device)

        if self.warm_start:
            self._load_state()

        optimizer = torch.optim.LBFGS(
            list(self.adjoint_model.parameters()),
            lr=self.lr,
            max_iter=self.max_iter,
            tolerance_grad=self.tol,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        info = {"final_loss": None, "niter": 0}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = self._residual_loss(y_vals)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        info["final_loss"] = float(loss.detach().cpu().item())

        with torch.no_grad():
            p_vals = self.adjoint_model(self.xy).detach()

        self._save_state()
        
        return p_vals, info


# =========================================================
# 6) Reduced objective
# =========================================================
class ReducedSemilinearOCObjective:
    """
    Reduced objective:
        j(theta_u) = 0.5 || y(theta_u) - g ||^2 + 0.5 alpha || u(theta_u) ||^2

    where y(theta_u) is obtained from the state solver.
    """

    def __init__(
        self,
        control_model: nn.Module,
        state_solver: WeakSemilinearStateSolver,
        adjoint_solver: WeakSemilinearAdjointSolver,
        xy: torch.Tensor,
        g_target: torch.Tensor,
        alpha: float,
        nonlinearity: nn.Module,
        weight=None,
        device="cpu",
        fd_eps_hess=1e-6,
        value_cold_start=True,
        grad_cold_start=False,
    ):
        self.control_model = control_model.to(device)
        self.state_solver = state_solver
        self.adjoint_solver = adjoint_solver
        self.xy = xy.to(device)
        self.g_target = g_target.to(device)
        self.alpha = float(alpha)
        self.nonlinearity = nonlinearity.to(device)
        self.device = device
        self.fd_eps_hess = float(fd_eps_hess)

        if weight is None:
            self.weight = None
        else:
            self.weight = weight.to(device) if torch.is_tensor(weight) else weight

        # Debug/stability options
        self.value_cold_start = bool(value_cold_start)
        self.grad_cold_start = bool(grad_cold_start)

        # caches
        self._last_u = None
        self._last_y = None
        self._last_p = None
        self._last_state_info = None
        self._last_adj_info = None

    def update(self, theta, flag: str):
        load_vector_into_model(theta, self.control_model)

    def _weighted_mean(self, z: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            return torch.mean(z)

        w = self.weight
        if torch.is_tensor(w) and w.numel() == 1:
            return w * torch.sum(z)

        if torch.is_tensor(w) and w.ndim == 1:
            w = w.reshape(-1, 1)

        return torch.sum(w * z)

    def _compute_control(self, theta):
        load_vector_into_model(theta, self.control_model)
        with torch.no_grad():
            u_vals = self.control_model(self.xy, smooth=False).detach()
        return u_vals

    def solve_state_only(self, theta, cold_start=None):
        if cold_start is None:
            cold_start = self.value_cold_start

        u_vals = self._compute_control(theta)

        if cold_start:
            self.state_solver.reset_state()

        y_vals, state_info = self.state_solver.solve(u_vals)

        self._last_u = u_vals
        self._last_y = y_vals
        self._last_state_info = state_info
        return y_vals, u_vals, state_info

    def solve_state_and_adjoint(self, theta, cold_start_state=None, cold_start_adjoint=None):
        if cold_start_state is None:
            cold_start_state = self.grad_cold_start
        if cold_start_adjoint is None:
            cold_start_adjoint = self.grad_cold_start

        u_vals = self._compute_control(theta)

        if cold_start_state:
            self.state_solver.reset_state()
        y_vals, state_info = self.state_solver.solve(u_vals)

        if cold_start_adjoint:
            self.adjoint_solver.reset_state()
        p_vals, adj_info = self.adjoint_solver.solve(y_vals)

        self._last_u = u_vals
        self._last_y = y_vals
        self._last_p = p_vals
        self._last_state_info = state_info
        self._last_adj_info = adj_info

        return y_vals, u_vals, p_vals, state_info, adj_info

    def value(self, theta, ftol=1e-12):
        # Important: value should not solve the adjoint
        y_vals, u_vals, _ = self.solve_state_only(theta, cold_start=self.value_cold_start)

        with torch.no_grad():
            tracking = 0.5 * self._weighted_mean((y_vals - self.g_target) ** 2)
            reg = 0.5 * self.alpha * self._weighted_mean(u_vals ** 2)
            val = tracking + reg

        return float(val.detach().cpu().item()), 0.0

    def value_model(self, theta, ftol=1e-12):
        return self.value(theta, ftol)

    def gradient(self, theta, gtol=1e-12):
        """
        Reduced gradient:
            grad_u j = alpha * u + p
        and then map to parameter space by chain rule.

        Important fix:
        detach the multiplier alpha*u + p before differentiating wrt theta,
        otherwise the alpha*u term gets differentiated twice.
        """
        y_vals, u_vals, p_vals, _, _ = self.solve_state_and_adjoint(
            theta,
            cold_start_state=self.grad_cold_start,
            cold_start_adjoint=self.grad_cold_start,
        )

        theta_dict = OrderedDict(
            (k, v.detach().clone().requires_grad_(True))
            for k, v in self.control_model.named_parameters()
        )

        u = functional_call(self.control_model, theta_dict, (self.xy, False))

        # Critical fix: detach multiplier
        grad_density = self.alpha * u_vals.detach() + p_vals.detach()

        # Scalar whose theta-gradient gives:
        #   (du/dtheta)^T (alpha*u + p)
        loss = self._weighted_mean(grad_density * u)

        grads = torch.autograd.grad(loss, tuple(theta_dict.values()), create_graph=False)
        grad_td = OrderedDict((k, g.detach().clone()) for k, g in zip(theta_dict.keys(), grads))
        return TorchDictVector(grad_td), 0.0

    def hessVec(self, v, theta, gradTol=1e-12):
        """
        Finite-difference Hessian-vector product.
        Expensive and noisy, but acceptable as a first version.
        """
        eps = self.fd_eps_hess

        theta_plus = theta.copy()
        theta_plus.axpy(eps, v)

        g_plus, _ = self.gradient(theta_plus, gradTol)
        g_base, _ = self.gradient(theta, gradTol)

        hv = (g_plus - g_base) * (1.0 / eps)
        return hv, 0.0

    def relative_L2_error_control(self, theta, u_star_fn):
        load_vector_into_model(theta, self.control_model)
        with torch.no_grad():
            u_pred = self.control_model(self.xy, smooth=False)
            u_true = u_star_fn(self.xy)
            return torch.sqrt(torch.mean((u_pred - u_true) ** 2) / torch.mean(u_true ** 2)).item()

    def relative_L2_error_state(self, theta, y_star_fn):
        y_vals, _, _ = self.solve_state_only(theta, cold_start=self.value_cold_start)
        with torch.no_grad():
            y_true = y_star_fn(self.xy)
            return torch.sqrt(torch.mean((y_vals - y_true) ** 2) / torch.mean(y_true ** 2)).item()
        
# =========================================================
# 7) Trust-region code
# =========================================================
def set_default_parameters(name):
    params = {}

    # General Parameters
    params['spsolver']  = name.replace(' ', '')
    params['outFreq']   = 1
    params['debug']     = False
    params['initProx']  = False
    params['t']         = 1
    params['safeguard'] = np.sqrt(np.finfo(float).eps)

    # Stopping tolerances
    params['maxit']   = 200
    params['reltol']  = False
    params['gtol']    = 9e-3
    params['stol']    = 1e-12
    params['ocScale'] = params['t']

    # Trust-region parameters
    params['eta1']     = 1e-4
    params['eta2']     = 0.75
    params['gamma1']   = 0.25
    params['gamma2']   = 1.5
    params['delta']    = 1
    params['deltamin'] = 1e-8

    params['deltamax'] = 1.0

    # Subproblem solve tolerances
    params['atol']    = 1e-5
    params['rtol']    = 1e-3
    params['spexp']   = 2
    params['maxitsp'] = 30

    # GCP and subproblem solve parameter
    params['useGCP']    = False
    params['mu1']       = 1e-4
    params['beta_dec']  = 0.1
    params['beta_inc']  = 10.0
    params['maxit_inc'] = 2

    # SPG and spectral GCP parameters
    params['lam_min'] = 1e-12
    params['lam_max'] = 1e12
    
    params["nonmono_M"] = 10

    
    return params

def compute_gradient(x, problem, params, cnt):
    """
    Compute the gradient with inexactness handling.

    Parameters:
    x (np.array): Current point.
    problem (Problem): Problem class.
    params (dict): Algorithm parameters.
    cnt (dict): Counters.

    Returns:
    grad (np.array): Gradient.
    dgrad (np.array): Dual gradient.
    gnorm (float): Gradient norm.
    cnt (dict): Updated counters.
    """
    use_inexact_now = bool(params.get("_useInexactGrad_now", params['useInexactGrad']))
    if use_inexact_now:
        scale0 = params['scaleGradTol']
        gtol = min(params['maxGradTol'], scale0 * params['delta'])
        gerr = gtol + 1
        while gerr > gtol:
            problem.obj_smooth.update(x,'lock')
            grad, gerr = problem.obj_smooth.gradient(x, gtol)
            cnt['ngrad'] += 1
            dgrad = problem.dvector.dual(grad)
            pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
            cnt['nprox'] += 1
            gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']
            gtol = min(params['maxGradTol'], scale0 * min(gnorm, params['delta']))
    else:
        gtol = 1e-12
        problem.obj_smooth.update(x,'lock')
        grad, gerr = problem.obj_smooth.gradient(x, gtol)
        cnt['ngrad'] += 1
        dgrad = problem.dvector.dual(grad)
        pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
        cnt['nprox'] += 1
        gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']

    params['gradTol'] = gtol
    cnt['graderr'].append(gerr)
    cnt['gradtol'].append(gtol)
    return grad, dgrad, gnorm, cnt 



    
def trustregion_gcp2(x,val,grad,dgrad,phi,problem,params,cnt):
  params.setdefault('safeguard', np.sqrt(np.finfo(float).eps))  # Numerical safeguard
  params.setdefault('lam_min', 1e-12)
  params.setdefault('lam_max', 1e12)
  params.setdefault('t', 1)
  params.setdefault('t_gcp', params['t'])
  params.setdefault('gradTol', np.sqrt(np.finfo(float).eps)) # Gradient inexactness tolerance used for hessVec

  ## Compute Cauchy point as a single SPG step
  Hg,_          = problem.obj_smooth.hessVec(grad,x,params['gradTol'])
  #cnt['nhess'] += 1
  gHg           = problem.dvector.apply(Hg, grad)
  gg            = problem.pvector.dot(grad, grad)
  if (gHg > params['safeguard'] * gg):
    t0Tmp = gg / gHg
  else:
    t0Tmp = params['t'] / np.sqrt(gg)
  t0     = np.min([params['lam_max'],np.max([params['lam_min'], t0Tmp])])
  xc     = problem.obj_nonsmooth.prox(x - t0 * dgrad, t0)
  cnt['nprox'] += 1
  s      = xc - x
  snorm  = problem.pvector.norm(s)
  Hs, _  = problem.obj_smooth.hessVec(s,x,params['gradTol'])
  #cnt['nhess'] += 1
  sHs    = problem.dvector.apply(Hs,s)
  gs     = problem.pvector.dot(grad,s)
  phinew = problem.obj_nonsmooth.value(xc)
  cnt['nobj2'] += 1
  alpha  = 1
  if (snorm >= (1-params['safeguard'])*params['delta']):
    alpha = np.minimum(1, params['delta']/snorm)

  if sHs > params['safeguard']: #*problem.pvector.dot(s,s):
    alpha = np.minimum(alpha,-(gs+phinew-phi)/sHs) #min(alpha,max(-(gs + phinew - phi), snorm^2 / t0)/sHs);

  if (alpha != 1):
    s      *= alpha
    snorm  *= alpha
    gs     *= alpha
    Hs     *= alpha
    sHs    *= alpha**2
    xc     = x + s
    phinew = problem.obj_nonsmooth.value(xc)
    cnt['nobj2'] += 1

  valnew = val + gs + 0.5*sHs
  pRed   = (val+phi)-(valnew+phinew)
  params['t_gcp'] = t0
  return s, snorm, pRed, phinew, Hs, cnt, params

def trustregion_step_SPG2(x, val,grad, dgrad, phi, problem, params, cnt):
    params.setdefault('maxitsp', 30)
    ## Cauchy point parameters
    params.setdefault('lam_min', 1e-12)
    params.setdefault('lam_max', 1e12)
    params.setdefault('t', 1)
    params.setdefault('gradTol', np.sqrt(np.finfo(float).eps))
    ## General parameters
    params.setdefault('safeguard', np.sqrt(np.finfo(float).eps))  # Numerical safeguard
    params.setdefault('atol',  1e-4) # Absolute tolerance
    params.setdefault('rtol',  1e-2) # Relative tolerance
    params.setdefault('spexp',    2) # hk0 exponent

    x0    = copy.deepcopy(x)
    g0_primal    = copy.deepcopy(grad)
    snorm = 0

    # Evaluate model at GCP
    sHs    = 0
    gs     = 0
    valold = val
    phiold = phi
    hk0    = 0
    valnew = valold
    phinew = phiold

    [sc,snormc,pRed,_,_,cnt,params] = trustregion_gcp2(x,val,grad,dgrad,phi,problem,params,cnt)

    t0     = params['t']
    s      = copy.deepcopy(sc)
    x1     = x0 + s
    gnorm  = snormc
    gtol   = np.min([params['atol'], params['rtol']*(gnorm/t0)**params['spexp']])

    # Set exit flag
    iter  = 0
    iflag = 1

    for iter0 in range(1, params['maxitsp'] + 1):
        alphamax = 1
        snorm0   = snorm
        snorm    = problem.pvector.norm(x1 - x)

        if snorm >= (1 - params['safeguard'])*params['delta']:
            ds = problem.pvector.dot(s, x0 - x)
            dd = gnorm**2
            alphamax = np.minimum(1, (-ds + np.sqrt(ds**2 + dd * (params['delta']**2 - snorm0**2)))/dd)
        Hs, _  = problem.obj_smooth.hessVec(s,x,params['gradTol'])
        #cnt['nhess'] += 1
        sHs    = problem.dvector.apply(Hs,s)
        #print("DEBUG:","||s||=", snorm,"sHs=", sHs,"||Hs||=", problem.pvector.norm(Hs),"gTs=", problem.pvector.dot(grad, s))
        g0s    = problem.pvector.dot(g0_primal,s)
        phinew = problem.obj_nonsmooth.value(x1)
        alpha0 = -(g0s + phinew - phiold) / sHs
        if (not np.isfinite(sHs)) or (sHs <= 1e-14):
            alpha = alphamax
        else:
            alpha = max(0.0, np.minimum(alphamax, alpha0))
        #if sHs <= params['safeguard']: 
        #  alpha = alphamax
        #else:
        #  alpha = np.minimum(alphamax,alpha0)
        ## Update iterate
        if (alpha == 1):
          x0     = x1
          g0_primal     += problem.dvector.dual(Hs)
          valnew = valold + g0s + 0.5 * sHs
        else:
          x0     += alpha*s
          g0_primal     += alpha*problem.dvector.dual(Hs)
          valnew = valold + alpha * g0s + 0.5 * alpha**2 * sHs
          phinew = problem.obj_nonsmooth.value(x0)
          snorm  = problem.pvector.norm(x0-x)

        ## Update model information
        valold = valnew
        phiold = phinew

        ## Check step size
        if snorm >= (1-params['safeguard'])*params['delta']:
          iflag = 2
          break

        # Update spectral step length
        if sHs <= params['safeguard']: #*gnorm**2:
          lambdaTmp = params['t']/problem.pvector.norm(g0_primal)
        else:
          lambdaTmp = gnorm**2/sHs

        t0 = np.max([params['lam_min'],np.min([params['lam_max'], lambdaTmp])])
        ## Compute step
        x1    = problem.obj_nonsmooth.prox(x0 - t0 * g0_primal, t0)
        s     = x1 - x0
        ## Check for convergence
        gnorm = problem.pvector.norm(s)
        if (gnorm/t0 <= gtol):
          iflag = 0
          break

    s    = x0 - x
    pRed = (val+phi) - (valnew+phinew)
    if (iter0 > iter):
       iter = iter0
    return s, snorm, pRed, phinew, iflag, iter, cnt, params

from collections import deque

def trustregion(x0, Deltai, problem, params):
    start_time = time.time()

    # ---------------- defaults ----------------
    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 500)
    params.setdefault('gtol', 9e-3)
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
    params.setdefault('deltamax', 1.0)
    params.setdefault('reltol', False)

    params.setdefault('delta_stop', 1e-7)
    params.setdefault('stol_abs', 1e-9)
    params.setdefault('stag_window', 10)
    params.setdefault('ftol_rel', 1e-6)
    params.setdefault('max_reject', 15)
    params.setdefault("nonmono_M", 10)

    # new: tiny predicted reduction termination
    params.setdefault("pred_abs_tol", 1e-8)
    params.setdefault("pred_rel_tol", 1e-8)
    params.setdefault("pred_small_max", 5)

    
    

    # ---------------- counters/history ----------------
    cnt = {
        'AlgType': f"TR-{params.get('spsolver','SPG2')}",
        'iter': 0,
        'nobj1': 0,
        'ngrad': 0,
        'nobj2': 0,
        'nprox': 0,
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
        'valerr': [],
        'valtol': [],
        'graderr': [],
        'gradtol': []
    }

    obj = problem.obj_smooth

    # ---------------- init x ----------------
    if params['initProx']:
        x = problem.obj_nonsmooth.prox(x0, 1.0)
        cnt['nprox'] += 1
    else:
        x = x0.copy()

    obj.update(x, "init")



    # ---------------- initial eval ----------------
    
    print("Initializing trust-region method...", flush=True)

    val_true, _ = obj.value(x, 1e-12)
    cnt['nobj1'] += 1
    

    if hasattr(obj, "value_model"):
        val_model, _ = obj.value_model(x, 1e-12)
        cnt['nobj1'] += 1
    else:
        val_model = val_true

    # set mode before gradient
    
    print("Initial objective computed. Now computing initial gradient...", flush=True)
    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
    

    phi = problem.obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    Facc = [val_true + phi]
    Fhist = deque(maxlen=params["nonmono_M"])
    Fhist.append(val_true + phi)

    # ---------------- header ----------------
    print(f"TR method using {params.get('spsolver','SPG2')} Subproblem Solver")
    print("  iter    value    gnorm    del    snorm    nobjs    ngrad    nobjn    nprox     iterSP    flagSP")
    print(f"{0:4d} {0:4d} {val_true+phi:8.6e} {params['delta']:8.6e}  ---      "
          f"{cnt['nobj1']:6d}      {cnt['ngrad']:6d}      {cnt['nobj2']:6d}      {cnt['nprox']:6d} --- ---")

    # ---------------- store init ----------------
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

    # ---------------- stopping tolerances ----------------
    gtol = params['gtol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm
    rej_count = 0
    small_pred_count = 0

    # ==========================
    # main loop
    # ==========================
    for i in range(1, params['maxit'] + 1):
        
        grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
    
        # model anchor
        if hasattr(obj, "value_model"):
            val_model, _ = obj.value_model(x, 1e-12)
        else:
            val_model = val_true
        cnt['nobj1'] += 1

        # solve TR subproblem
        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
            x, val_model, grad, dgrad, phi, problem, params, cnt
        )

        pRed = float(pRed)
        pred_floor = max(
            params["pred_abs_tol"],
            params["pred_rel_tol"] * max(1.0, abs(val_model + phi))
        )

        # new: tiny predicted reduction logic
        if pRed <= pred_floor:
            small_pred_count += 1
        else:
            small_pred_count = 0

        if (small_pred_count >= params["pred_small_max"]) and (params["delta"] <= 10.0 * params["delta_stop"]):
            cnt['iter'] = i
            cnt['timetotal'] = time.time() - start_time
            cnt['iflag'] = 6
            print("Optimization terminated because predicted reduction is tiny repeatedly.")
            print(f"Total time: {cnt['timetotal']:8.6e} seconds")
            return x, cnt

        # trial point
        xnew = x + s

        # true acceptance objective/value
        valnew_true, _ = obj.value(xnew, 1e-12)
        cnt['nobj1'] += 1
        phinew_true = problem.obj_nonsmooth.value(xnew)
        cnt['nobj2'] += 1

        aRed = (val_true + phi) - (valnew_true + phinew_true)

        rho = -np.inf if pRed <= 0.0 else float(aRed) / pRed
        Fref = max(Fhist)
        accept_nm = (valnew_true + phinew_true) <= (Fref - 1e-12)
        accept = (rho >= params['eta1']) and accept_nm

        print("debug:",
              "aRed=", float(aRed),
              "pRed=", float(pRed),
              "rho=", float(rho))

        if not accept:
            params['delta'] = max(params['deltamin'], params['gamma1'] * params['delta'])
            obj.update(x, 'reject')
            rej_count += 1

        else:
            # accept
            x = xnew
            phi = phinew_true
            val_true = valnew_true
            rej_count = 0

            obj.update(x, 'accept')

            Facc.append(val_true + phi)
            Fhist.append(val_true + phi)
            if hasattr(obj,"relative_L2_error_control"):
                rel_u = obj.relative_L2_error_control(x, lambda xy: u_star_oc(xy, obj.nonlinearity))
                rel_y = obj.relative_L2_error_state(x, y_star_oc)
                
                print("relative L2 error(control)=", rel_u)
                print("relative L2 state   =", rel_y)


            if rho > params['eta2']:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])

        # print iteration line
        if i % params['outFreq'] == 0:
            print(f"{i:4d}    {val_true + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      "
                  f"{cnt['nobj1']:6d}     {cnt['ngrad']:6d}       {cnt['nobj2']:6d}     {cnt['nprox']:6d}      "
                  f"{iter_count:4d}        {iflag:1d}")

        # history
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

        # ---------------- stopping tests ----------------
        delta_stop = params["delta_stop"]
        stol_abs   = params["stol_abs"]
        K          = params["stag_window"]
        ftol_rel   = params["ftol_rel"]
        max_reject = params["max_reject"]

        stop_grad = (gnorm <= gtol)
        stop_step = (snorm < stol_abs) and (params["delta"] <= delta_stop)
        stop_stuck = (params["delta"] <= 10 * delta_stop and rej_count >= max_reject)

        stop_stag = False
        if len(Facc) >= K + 1:
            Fold = Facc[-(K+1)]
            Fnew = Facc[-1]
            rel_change = abs(Fold - Fnew) / max(1.0, abs(Fnew))
            stop_stag = (rel_change < ftol_rel)

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
                reason = "trust region stuck (delta small + rejections)"
            else:
                flag = 1
                reason = "maximum iterations reached"

            cnt['iter'] = i
            cnt['timetotal'] = time.time() - start_time
            cnt['iflag'] = flag

            print("Optimization terminated because", reason)
            print(f"Total time: {cnt['timetotal']:8.6e} seconds")
            return x, cnt

    # fallback
    cnt['iter'] = params['maxit']
    cnt['timetotal'] = time.time() - start_time
    cnt['iflag'] = 1
    return x, cnt

# ===========================================
# 8) Training evaluation plotting
# ===========================================


def train_semilinear_oc_reduced_TR(
    width_state=32,
    depth_state=2,
    width_control=32,
    depth_control=2,
    width_adjoint=32,
    depth_adjoint=2,
    ngrid=12,
    alpha=1e-2,
    beta_l1=0.0,
    delta0=1e-1,
    maxit=50,
    device="cpu",
    seed=None,
    mmax=8,
    nmax=8,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    xy = make_training_points_grid(ngrid, device=device)
    modes = [(i, j) for i in range(1, mmax + 1) for j in range(1, nmax + 1)]
    V, dV = make_sine_tests(xy, modes=modes)

    nonlinearity = MonotoneReLUlinearity(
        c0=1.0, a=(0.5, 0.25), b=(1.0, 2.0), c=(0.0, -0.2)
    ).to(device)

    g_target = g_target_from_optimality(xy, nonlinearity, alpha)

    control_model = ControlNet(
        in_dim=2, width=width_control, depth=depth_control, activation="relu"
    ).to(device)

    state_model = StateNet(
        in_dim=2, width=width_state, depth=depth_state, activation="relu"
    ).to(device)

    adjoint_model = AdjointNet(
        in_dim=2, width=width_adjoint, depth=depth_adjoint, activation="relu"
    ).to(device)

    state_solver = WeakSemilinearStateSolver(
        state_model=state_model,
        nonlinearity=nonlinearity,
        xy=xy,
        V=V,
        dV=dV,
        device=device,
        max_iter=100,
        tol=1e-10,
        lr=1.0,
        warm_start=True,
    )

    adjoint_solver = WeakSemilinearAdjointSolver(
        adjoint_model=adjoint_model,
        nonlinearity=nonlinearity,
        xy=xy,
        V=V,
        dV=dV,
        g_target=g_target,
        device=device,
        max_iter=100,
        tol=1e-10,
        lr=1.0,
        warm_start=True,
    )

    obj_smooth = ReducedSemilinearOCObjective(
        control_model=control_model,
        state_solver=state_solver,
        adjoint_solver=adjoint_solver,
        xy=xy,
        g_target=g_target,
        alpha=alpha,
        nonlinearity=nonlinearity,
        device=device,
        fd_eps_hess=1e-6,
        value_cold_start=True,
        grad_cold_start=False,
    )

    var = {"useEuclidean": False, "beta": beta_l1}
    

    class _ProblemWrap:
        pass

    problem = _ProblemWrap()
    problem.obj_smooth = obj_smooth
    problem.obj_nonsmooth = ControlOnlyL1TorchNorm(var)
    problem.pvector = L2TVPrimal(var)
    problem.dvector = L2TVDual(var)

    x0 = vector_from_model(control_model)

    params = set_default_parameters("SPG2")
    params["delta"] = delta0
    params["maxit"] = maxit
    params["useInexactObj"] = False
    params["useInexactGrad"] = False
    

    x_opt, cnt = trustregion(x0, delta0, problem, params)

    load_vector_into_model(x_opt, control_model)
    return control_model, state_model, adjoint_model, x_opt, cnt, problem
    

def evaluate_oc_solution_reduced(problem, theta):
    obj = problem.obj_smooth
    y_vals, u_vals, p_vals, s_info, a_info = obj.solve_state_and_adjoint(theta)
    with torch.enable_grad():
        tracking = 0.5 * torch.mean((y_vals-obj.g_target)**2)
        reg = 0.5 * obj.alpha * torch.mean(u_vals**2)
        val = tracking +reg
        rel_state = torch.sqrt(torch.mean((y_vals-y_star_oc(obj.xy))**2)/
                                          torch.mean(y_star_oc(obj.xy)**2)).item()
        rel_control = torch.sqrt(torch.mean((u_vals-u_star_oc(obj.xy,obj.nonlinearity))**2)/
                                          torch.mean(u_star_oc(obj.xy,obj.nonlinearity)**2)).item()
        

    
        
        
    
    print("\nObjective decomposition:")
    print("tracking term  =", tracking.item())
    print("control reg    =", reg.item())
    print("total objective=", val.item())
    print("state solver loss=", s_info["final_loss"])
    print("adjoint solver loss=", a_info["final_loss"])

    

    return {
        "objective_penalized": float(val.item()),
        "rel_state_L2": rel_state,
        "rel_control_L2": rel_control,
        "u_norm": float(torch.sqrt(torch.mean(u_vals ** 2)).item()),
        "y_norm": float(torch.sqrt(torch.mean(y_vals ** 2)).item()),
        "p_norm": float(torch.sqrt(torch.mean(p_vals ** 2)).item()),
    }

def plot_oc_results_reduced(problem, theta, ngrid_plot=None):
    
    obj = problem.obj_smooth
    y_vals, u_vals, _, _, _ = obj.solve_state_and_adjoint(theta)


    xy = obj.xy
    if ngrid_plot is None:
        ngrid_plot = int(round(np.sqrt(xy.shape[0])))

    with torch.no_grad():
        y_pred = y_vals.detach().cpu().numpy().reshape(ngrid_plot, ngrid_plot)
        u_pred = u_vals.detach().cpu().numpy().reshape(ngrid_plot, ngrid_plot)

        y_true = y_star_oc(xy).detach().cpu().numpy().reshape(ngrid_plot, ngrid_plot)
        u_true = u_star_oc(xy, obj.nonlinearity).detach().cpu().numpy().reshape(ngrid_plot, ngrid_plot)

    err_y = y_pred - y_true
    err_u = u_pred - u_true

    x = xy[:, 0].detach().cpu().numpy().reshape(ngrid_plot, ngrid_plot)
    y = xy[:, 1].detach().cpu().numpy().reshape(ngrid_plot, ngrid_plot)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    fields = [
        (y_pred, r"Predicted state $y_h$"),
        (y_true, r"Exact state $y^*$"),
        (err_y,  r"State error $y_h-y^*$"),
        (u_pred, r"Predicted control $u_h$"),
        (u_true, r"Exact control $u^*$"),
        (err_u,  r"Control error $u_h-u^*$"),
    ]

    for ax, (Z, title) in zip(axes.flat, fields):
        im = ax.contourf(x, y, Z, levels=50)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)

    plt.show()
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    control_model, state_model, adjoint_model, x_opt, cnt, problem=train_semilinear_oc_reduced_TR(
        width_state = 24,
        depth_state = 2,
        width_control = 24,
        depth_control = 2,
        width_adjoint = 24,
        depth_adjoint = 2,
        ngrid = 10,
        alpha = 1e-2,
        beta_l1 = 0.0,
        delta0 = 1e-1,
        maxit = 80,
        device = device,
        seed = 0,
        mmax = 6,
        nmax = 6,
        )
    
    stats = evaluate_oc_solution_reduced(problem, x_opt)
    print("\nFinal OC stats:")
    for k,v in stats.items():
        print(f"{k}:{v:.6e}" if isinstance(v,float) else f"{k}:{v}")
    
    plot_oc_results_reduced(problem, x_opt)  
