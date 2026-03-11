import torch
import numpy as np
import copy,time
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp, vjp

# -------------------------
# 0) Fourier Features
# -------------------------


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
    
    @torch.no_grad()
    def min_abs_preact(self, x: torch.Tensor) -> float:
        return self.mlp.min_abs_preact(self.ff(x))
    @torch.no_grad()
    def kink_score(self,x,q=0.01):
        return self.mlp.kink_score(self.ff(x),q=q)
    @torch.no_grad()
    def softplus_relu_grad_gap(self, x: torch.Tensor, q: float = 0.90) -> float:
        return self.mlp.softplus_relu_grad_gap(self.ff(x), q=q)
    
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
                h = F.softplus(h,beta = beta)/beta
            else:
                h = layer(h)
        
        return h
    
    @torch.no_grad()
    def softplus_relu_grad_gap(self, x: torch.Tensor, q: float=0.90) -> float:
        """
        Measure | sigmoid(beta * a) - 1_{a>0} | over all ReLU preactivations a.
        Returns a quantile of the mismatch distribution.
        
        Smaller means softplus gradient is closer to ReLU gradient.
        
        """
        if self.activation != "relu":
            return 0.0
        
        h = x
        beta = float(self.smooth_beta)
        gaps = []
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                h = layer(h)
            elif isinstance(layer, nn.ReLU):
                relu_grad = (h>0).to(h.dtype)
                soft_grad = torch.sigmoid(beta * h)
                gaps.append((soft_grad - relu_grad).abs().reshape(-1))
                h = torch.relu(h)
            else:
                h = layer(h)
        if not gaps:
            return 0.0
        
        g = torch.cat(gaps)
        return float(torch.quantile(g,q).cpu().item())
    
    @torch.no_grad()
    def min_abs_preact(self, x: torch.Tensor) -> float:
        """
        Return min |preactivation| over all ReLU layers and all samples.
        preactivation = input to ReLU (i.e., output of Linear before ReLU).
        """
        h = x
        m = float("inf")
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                h = layer(h)
            elif isinstance(layer, nn.ReLU):
                m = min(m, float(h.abs().min().cpu().item()))
                h = torch.relu(h)
            else:
                h = layer(h)
        return m
    @torch.no_grad()
    def kink_score(self, x, q=0.01):
        h = x
        vals = []
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                h = layer(h)
            elif isinstance(layer, nn.ReLU):
                vals.append(h.abs().reshape(-1))
                h = torch.relu(h)
            else:
                h = layer(h)
        if not vals:
            return float("inf")
        v = torch.cat(vals)
        return float(torch.quantile(v, q).cpu().item())


# -------------------------
# 1) Parameters Setups
# -------------------------


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

# Helpers for TorchDictVector
def td_dot(x: TorchDictVector, y: TorchDictVector) -> float:
    s = 0.0
    for k,v in x.td.items():
        s = s + torch.sum(v*y.td[k]).item()
    return float(s)

def td_axpy(a: float, x: TorchDictVector, y: TorchDictVector) -> TorchDictVector:
    # return a*x+y
    out = TorchDictVector()
    for k in x.td.keys():
        out.td[k] = a * x.td[k] + y.td[k]
    return out

def td_scale(a: float, x: TorchDictVector) -> TorchDictVector:
    out = TorchDictVector()
    for k, v in x.td.items():
        out.td[k] = a*v
    return out

def vector_from_model(model: torch.nn.Module) -> TorchDictVector:
    td = OrderedDict()
    for name, p in model.named_parameters():
        td[name] = p.detach().clone()
    return TorchDictVector(td)

#@torch.no_grad()
def load_vector_into_model(x: TorchDictVector, model: torch.nn.Module):
    with torch.no_grad():
        name_to_param = dict(model.named_parameters())
        for k, v in x.td.items():
            name_to_param[k].copy_(v)

def ensure_same_structure(x: TorchDictVector, model: torch.nn.Module):
    model_keys = [n for n, _ in model.named_parameters()]
    x_keys = list(x.td.keys())
    assert model_keys == x_keys, "TorchDictVector keys must match model.named_parameters() order."
    
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


    # Debug / diagnostics (all optional; default off)
    params.setdefault('debug_drop_gate', False)      # print drop-gate info each iter
    params.setdefault('debug_h_equiv', False)        # print h-equivalence diagnostic
    params.setdefault('debug_h_equiv_freq', 1)       # how often to print (every k iters)

    # Numerical guards/tolerances
    params.setdefault('prox_equiv_abs_tol', 1e-10)   # tight-frame prox identity tolerance
    params.setdefault('min_drop_cap', 1e-8)          # min parent cap to allow dropping
    
    params["nonmono_M"] = 10  #add for relu
    
    # For stochastic sampling
    params.setdefault('batch_size_model', 512)
    params.setdefault("batch_size_acc",   4096)
    params.setdefault("batch_replace",    False)
    params.setdefault("batch_seed",       None) 
    
    params.setdefault("generator", None)
    
    # Hybrid smooth-switch controls
    params.setdefault('kink_tau', 5e-3)
    params.setdefault('kink_hyst', 5.0)
    # extra gate: require softplus slope ~= ReLU slope
    params.setdefault('grad_match_tol', 0.49)
    params.setdefault('auto_inexact_grad', True)
    params.setdefault('mu_smooth', 1e-4)
    params.setdefault('grad_match_q', 0.9)
    
    return params


# -------------------------
# 1) Objective Function
# -------------------------
# --- boundary factor for homogeneous Dirichlet ---
def b_factor(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return x*(1-x)*y*(1-y)

def u_star(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    #base = x*(1-x)*y*(1-y)
    #return (1 + 25*torch.sin(2*math.pi*x)*torch.sin(2*math.pi*y) + 10.0*x*y)
    return torch.sin(math.pi*x)*torch.sin(math.pi*y)

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

class PoissonWeakPINNObjective:
    """
    Weak-form PINN / VPINN objective for Poisson:
        J(theta) = 0.5 * sum_k R_k(theta)^2 + boundary penalty

    where
        R_k(theta) = ∫ kappa grad(u_theta)·grad(v_k) dx - ∫ g v_k dx

    This only uses first derivatives of u_theta, so ReLU is compatible.
    """
    def __init__(self, model, xy, g, kappa_fn, V, dV, weight=None, device="cpu",
                 mu_I=0.0, xb=None, wb=None, bc_target=None, lam_bc=0.0, 
                 kink_tau=5e-3,kink_hyst=5.0,_smooth_mode= False):
        self.model = model.to(device)
        self.xy = xy.to(device)
        self.g = g.to(device)
        self.kappa_fn = kappa_fn
        self.V = V.to(device)      # (N,K)
        self.dV = dV.to(device)    # (N,K,2)
        self.device = device

        self.weight = weight.to(device) if torch.is_tensor(weight) else weight
        self.mu_I = float(mu_I)

        self.xb = xb.to(device) if xb is not None else None
        self.wb = wb.to(device) if wb is not None else None
        self.bc_target = bc_target.to(device) if bc_target is not None else None
        self.lam_bc = float(lam_bc)

        self.xy_full = xy.detach().clone()
        self.g_full = g.detach().clone()
        self.weight_full = self.weight
        self.V_full = V.detach().clone()
        self.dV_full = dV.detach().clone()
        
        self.kink_tau = float(kink_tau)
        self.kink_hyst = float(kink_hyst)
        self._smooth_mode = bool(_smooth_mode)
        self._force_true = False

    def set_mu_I(self, mu_I: float):
        self.mu_I = float(mu_I)

    @torch.no_grad()
    def _set_parameters(self, theta):
        name_to_param = dict(self.model.named_parameters())
        for k, v in theta.td.items():
            name_to_param[k].copy_(v)
            
    @torch.no_grad()
    def decide_smooth_mode(self) -> bool:
        if getattr(self, "_force_true", False):
            self._smooth_mode = False
            return self._smooth_mode
        if not hasattr(self.model, "kink_score") and not hasattr(self.model, "min_abs_preact"):
            self._smooth_mode = False
            return self._smooth_mode
        
        vals = []
        if hasattr(self.model, "kink_score"):
            vals.append(float(self.model.kink_score(self.xy_full, q=0.01)))
            if self.xb is not None:
                vals.append(float(self.model.kink_score(self.xb,q=0.01)))
        else:
            vals.append(float(self.model.min_abs_preact(self.xy_full)))
            if self.xb is not None:
                vals.append(float(self.model.min_abs_preact(self.xy_full)))
        
        m = min(vals)
        if not self._smooth_mode:
            if m < self.kink_tau:
                self._smooth_mode = True
        else:
            if m > self.kink_hyst * self.kink_tau:
                self._smooth_mode = False
        return self._smooth_mode
                
        

    def update(self, theta, flag: str):
        self._set_parameters(theta)
        self._smooth_mode = False
        #if flag in ("init", "accept", "lock", "trial", "model"):
        #    self.decide_smooth_mode()

    def snapshot_batch(self):
        return (self.xy, self.g, self.weight, self.V, self.dV)

    def restore_batch(self, snap):
        self.xy, self.g, self.weight, self.V, self.dV = snap

    def set_batch(self, xy, g=None, weight=None, V=None, dV=None):
        if xy is not None:
            self.xy = xy
        if g is not None:
            self.g = g
        if weight is not None:
            self.weight = weight
        if V is not None:
            self.V = V
        if dV is not None:
            self.dV = dV
            
    def _model_forward(self, x, smooth=False):
        return self.model(x, smooth=False) if "smooth" in self.model.forward.__code__.co_varnames else self.model(x)

    #def _model_forward(self, x, smooth=False):
    #    if "smooth" in self.model.forward.__code__.co_varnames:
    #        return self.model(x,smooth=bool("smooth"))
    #    return self.model(x)

    def weak_residual_vector(self, theta, smooth=False):
        self._set_parameters(theta)

        xy = self.xy.detach().clone().requires_grad_(True)
        u = self._model_forward(xy,smooth=smooth)
        grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]   # (N,2)

        kap = self.kappa_fn(xy)
        if kap.ndim == 1:
            kap = kap.reshape(-1, 1)

        # integrand for each test function:
        #   kappa * grad_u · grad_v_k  -  g * v_k
        # shapes:
        #   grad_u      : (N,2)
        #   dV          : (N,K,2)
        #   V           : (N,K)
        gu_dot_dvk = (grad_u.unsqueeze(1) * self.dV).sum(dim=-1)   # (N,K)
        integrand = kap * gu_dot_dvk - self.g * self.V             # (N,K)

        if self.weight is None:
            R = integrand.mean(dim=0)   # (K,)
        else:
            w = self.weight
            if w.numel() == 1:
                R = (w * integrand).sum(dim=0)
            else:
                if w.ndim == 1:
                    w = w.reshape(-1, 1)
                R = (w * integrand).sum(dim=0)

        return R   # (K,)

    def _boundary_loss(self, theta, smooth=False):
        if (self.xb is None) or (self.lam_bc <= 0.0):
            return torch.zeros((), device=self.device)

        ub = self._model_forward(self.xb, smooth=smooth)
        target = 0.0 if self.bc_target is None else self.bc_target
        diff = ub - target

        if self.wb is None:
            return 0.5 * self.lam_bc * (diff**2).mean()

        w = self.wb.reshape(-1, 1) if self.wb.ndim == 1 else self.wb
        return 0.5 * self.lam_bc * (w * diff**2).sum()

    def value(self, theta, ftol=1e-12):
        with torch.enable_grad():
            R = self.weak_residual_vector(theta,smooth=False)
            val = 0.5 * torch.mean(R**2)
            val = val + self._boundary_loss(theta,smooth=False)
        return float(val.detach().cpu().item()), 0.0

    def value_model(self, theta, ftol=1e-12):
        with torch.enable_grad():
            R = self.weak_residual_vector(theta,smooth=bool(self._smooth_mode))
            val = 0.5*torch.mean(R**2)
            val = val + self._boundary_loss(theta,smooth=bool(self._smooth_mode))
            
        return float(val.detach().cpu().item()), 0.0

    def gradient(self, theta, gtol=1e-12):
        self._set_parameters(theta)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            R = self.weak_residual_vector(theta,smooth=bool(self._smooth_mode))
            val = 0.5 * torch.mean(R**2)
            val = val + self._boundary_loss(theta,smooth=bool(self._smooth_mode))
            val.backward()

        grad_td = OrderedDict(
            (name, p.grad.detach().clone())
            for name, p in self.model.named_parameters()
        )
        return TorchDictVector(grad_td), 0.0

    def hessVec(self, v, theta, gradTol=1e-12, eps_base=3e-4):
        vn = max(v.norm(), 1e-16)
        eps = eps_base/vn
        
        th_p = theta.copy()
        th_m = theta.copy()
        th_p.axpy(+eps, v)
        th_m.axpy(-eps, v)

        self.update(th_p, "lock")
        gp, _ = self.gradient(th_p, gradTol)

        self.update(th_m, "lock")
        gm, _ = self.gradient(th_m, gradTol)

        hv = gp.copy()
        hv.axpy(-1.0, gm)
        hv.scal(1.0 / (2.0 * eps))

        if self.mu_I != 0.0:
            hv.axpy(self.mu_I, v)

        self.update(theta, "lock")
        return hv, 0.0

    def relative_L2_error(self, theta):
        self._set_parameters(theta)
        with torch.no_grad():
            xy = self.xy_full
            u_pred = self.model(xy)
            u_true = u_star(xy)

            err_sq = torch.sum((u_pred - u_true) ** 2)
            true_sq = torch.sum(u_true ** 2)
            return torch.sqrt(err_sq / true_sq).item()
        
# -------------------------
# 3) Problem Class
# -------------------------

class L1TorchNorm:
    """
    phi(x) = beta * ||x||_1 for TorchDictVector x (dict of tensors)

    Required by your framework:
      - value(x) -> float
      - prox(x, t) -> TorchDictVector
    """
    def __init__(self, var):
        self.var = var  # expects var["beta"]

    @torch.no_grad()
    def value(self, x):
        beta = float(self.var["beta"])
        s = 0.0
        for v in x.td.values():
            s += torch.sum(torch.abs(v)).item()
        return beta * float(s)

    @torch.no_grad()
    def prox(self, x, t):
        """
        prox_{t*beta*||.||_1}(x) = sign(x) * max(|x| - t*beta, 0)
        """
        beta = float(self.var["beta"])
        tau = t * beta
        out = x.copy()  
        for k, v in x.td.items():
            out.td[k] = torch.sign(v) * torch.clamp(torch.abs(v) - tau, min=0.0)
        return out

    @torch.no_grad()
    def subgrad(self, x):
        """
        Returns one choice of subgradient in dict-form:
          beta*sign(x) with sign(0)=0
        (This is a valid element of the subdifferential.)
        """
        beta = float(self.var["beta"])
        out = OrderedDict()
        for k, v in x.td.items():
            out[k] = beta * torch.sign(v)
        return TorchDictVector(out)

    @torch.no_grad()
    def project_subgrad(self, g, x):
        """
        Project g onto subdifferential of beta||x||_1 (dict-form).

        For each entry:
          if x_i != 0: (proj) = beta*sign(x_i)
          if x_i == 0: (proj) = clamp(g_i, -beta, beta)
        """
        beta = float(self.var["beta"])
        out = OrderedDict()
        for k in x.td.keys():
            xv = x.td[k]
            gv = g.td[k]
            out[k] = torch.where(
                xv != 0,
                beta * torch.sign(xv),
                torch.clamp(gv, min=-beta, max=beta),
            )
        return TorchDictVector(out)

    def get_parameter(self):
        return float(self.var["beta"])
    

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


class Problem:
    """
    Container used by TR framework.

    Required fields used in your code:
      - obj_smooth: has update/value/gradient/hessVec
      - obj_nonsmooth: has value/prox
      - pvector: has dot/norm
      - dvector: has apply/dual
    """
    def __init__(self, obj_smooth, obj_nonsmooth, var=None):
        self.var = {} if var is None else dict(var)

        self.obj_smooth = obj_smooth
        self.obj_nonsmooth = obj_nonsmooth

        use_euclid = bool(self.var.get("useEuclidean", False))

        if use_euclid:
            self.pvector = DictEuclidean(self.var)
            self.dvector = DictEuclidean(self.var)
        else:
            
            self.pvector = L2TVPrimal(self.var)
            self.dvector = L2TVDual(self.var)
            
            
            
# -------------------------
# 3) Trust Region
# -------------------------
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
def should_use_inexact_gradient(obj, params) -> bool:
    """
    Turn on inexact/smoothed gradient only when:
      1) we are near a kink, and
      2) the smoothing gradient is close enough to the ReLU gradient.

    If the gradient-match diagnostic is unavailable, this falls back to just
    the kink test.
    """
    if not hasattr(obj, "decide_smooth_model"):
        return False
    if not hasattr(obj.model,"kink_score") and not hasattr(obj.model,"min_abs_preact"):
        return False

    xy_check = getattr(obj, "xy_full", obj.xy)

    # --- kink proximity ---
    #kink_close = False
    kink_tau = float(params.get("kink_tau", 5e-3))

    if hasattr(obj.model, "kink_score"):
        vals = [float(obj.model.kink_score(xy_check, q=0.01))]
        if getattr(obj, "xb", None) is not None:
            vals.append(float(obj.model.kink_score(obj.xb, q=0.01)))
        kink_measure = min(vals)
        #kink_close = (kink_measure < kink_tau)

    else: 
        vals = [float(obj.model.min_abs_preact(xy_check))]
        if getattr(obj, "xb", None) is not None:
            vals.append(float(obj.model.min_abs_preact(obj.xb)))
        kink_measure = min(vals)
        #kink_close = (kink_measure < kink_tau)
    has_grad_match = hasattr(obj.model, "softplus_relu_grad_gap") or hasattr(obj.model, "grad_match_score")
    if not has_grad_match:
        return False

    # --- smooth-vs-ReLU gradient agreement ---
    #grad_match = True
    grad_match_tol = float(params.get("grad_match_tol", 0.49))
    grad_match_q = float(params.get("grad_match_q", 0.90))

    if hasattr(obj.model, "softplus_relu_grad_gap"):
        vals = [float(obj.model.softplus_relu_grad_gap(xy_check, q=grad_match_q))]
        if getattr(obj, "xb", None) is not None:
            vals.append(float(obj.model.softplus_relu_grad_gap(obj.xb, q=grad_match_q)))
        grad_gap = max(vals)
        #grad_match = (grad_gap < grad_match_tol)

    else: 
        vals = [float(obj.model.grad_match_score(xy_check, q=grad_match_q))]
        if getattr(obj, "xb", None) is not None:
            vals.append(float(obj.model.grad_match_score(obj.xb, q=grad_match_q)))
        grad_gap = max(vals)
        #grad_match = (grad_gap < grad_match_tol)

    return bool(kink_measure < kink_tau) and (grad_gap < grad_match_tol)

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

    # new: auto inexact-gradient switch near kinks
    params.setdefault("useInexactGrad", False)          # global default
    params.setdefault("auto_inexact_grad", False)        # auto enable near kink
    params.setdefault("scaleGradTol", 1e-2)
    params.setdefault("maxGradTol", 1e-3)
    params.setdefault("kink_tau", 5e-3)
    params.setdefault("grad_match_tol", 0.49)
    params.setdefault("grad_match_q", 0.90)

    # optional regularization when smooth model is active
    params.setdefault("mu_smooth", 1e-4)
    

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

    # store full data once
    obj.xy_full = obj.xy
    if hasattr(obj, "g"):
        obj.g_full = obj.g
    obj.weight_full = getattr(obj, "weight", None)
    if hasattr(obj, "V"):
        obj.V_full = obj.V
    if hasattr(obj, "dV"):
        obj.dV_full = obj.dV

    # ---------------- helper for gradient mode ----------------
    def set_gradient_mode(x_cur):
        obj.update(x_cur, "lock")

        use_inexact = bool(params.get("useInexactGrad", False))
        if params.get("auto_inexact_grad", True):
            use_inexact = use_inexact or should_use_inexact_gradient(obj, params)

        params["_useInexactGrad_now"] = use_inexact

        # regularization only when smooth/inexact mode is active
        if hasattr(obj, "set_mu_I"):
            if use_inexact and getattr(obj, "_smooth_mode", False):
                obj.set_mu_I(params["mu_smooth"])
            else:
                obj.set_mu_I(0.0)

        return use_inexact

    # ---------------- initial eval ----------------
    rej_count = 0
    small_pred_count = 0

    val_true, _ = obj.value(x, 1e-12)
    cnt['nobj1'] += 1

    if hasattr(obj, "value_model"):
        val_model, _ = obj.value_model(x, 1e-12)
        cnt['nobj1'] += 1
    else:
        val_model = val_true

    # set mode before gradient
    use_inexact_now = set_gradient_mode(x)

    # temporarily override compute_gradient behavior
    old_use_inexact = params.get("useInexactGrad", False)
    params["useInexactGrad"] = bool(use_inexact_now)
    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
    params["useInexactGrad"] = old_use_inexact

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
    stol = params['stol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm
        stol = params['stol'] * gnorm

    # ==========================
    # main loop
    # ==========================
    for i in range(1, params['maxit'] + 1):

        # choose derivative mode at current iterate
        use_inexact_now = set_gradient_mode(x)

        # gradient for subproblem
        old_use_inexact = params.get("useInexactGrad", False)
        params["useInexactGrad"] = bool(use_inexact_now)
        params['tolsp'] = min(params['atol'], params['rtol'] * (gnorm ** params['spexp']))
        grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
        params["useInexactGrad"] = old_use_inexact

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
              "rho=", float(rho),
              "useInexactGrad=", bool(use_inexact_now),
              "smooth_mode=", bool(getattr(obj, "_smooth_mode", False)))

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

            relL2 = obj.relative_L2_error(x)
            print("relative L2 error =", relL2)
            print("boundary L2 =", boundary_L2(obj.model, obj.xb, device))

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

@torch.no_grad()
def boundary_L2(model, xb, device="cpu"):
    if "smooth" in model.forward.__code__.co_varnames:
        ub = model(xb.to(device), smooth=False)
    else:
        ub = model(xb.to(device))
    return torch.sqrt(torch.mean(ub**2)).item()



def compute_value(x, xprev, fvalprev, obj, pRed, params, cnt):
    """
    Compute the objective function value with inexactness handling.

    Parameters:
    x (np.array): Current point.
    xprev (np.array): Previous point.
    fvalprev (float): Previous function value.
    obj (Objective): Objective function class.
    pRed (float): Predicted reduction.
    params (dict): Algorithm parameters.
    cnt (dict): Counters.

    Returns:
    fval (float): Current function value.
    fvalprev (float): Previous function value.
    cnt (dict): Updated counters.
    """
    ftol = 1e-12
    valerrprev = 0
    if params['useInexactObj']:
        omega = params['expValTol']
        scale = params['scaleValTol']
        force = params['forceValTol']
        eta = params['etascale'] * min(params['eta1'], 1 - params['eta2'])
        ftol = min(params['maxValTol'], scale * (eta * min(pRed, force ** cnt['nobj1'])) ** (1 / omega))
        obj.update(xprev, "accept") 
        fvalprev, valerrprev = obj.value(xprev, ftol)
        cnt['nobj1'] += 1

    obj.update(x, 'trial')
    fval, valerr = obj.value(x, ftol)
    cnt['nobj1'] += 1
    cnt['valerr'].append(max(valerr, valerrprev))
    cnt['valtol'].append(ftol)

    return fval, fvalprev, cnt

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

# -------------------------
# 4) Training the model
# -------------------------


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

def make_boundary_points(n=32, device="cpu", dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    t = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)

    # 4 edges: (0,t), (1,t), (t,0), (t,1)
    left   = torch.stack([torch.zeros_like(t), t], dim=1)
    right  = torch.stack([torch.ones_like(t),  t], dim=1)
    bottom = torch.stack([t, torch.zeros_like(t)], dim=1)
    top    = torch.stack([t, torch.ones_like(t)], dim=1)

    xb = torch.cat([left, right, bottom, top], dim=0)  # (4n,2)

    # crude trapezoid-ish edge weight: each edge length=1, n points => ds ~ 1/(n-1)
    # total boundary length is 4, so sum weights ~4
    ds = 1.0 / max(1, (n - 1))
    wb = torch.full((xb.shape[0], 1), ds, device=device, dtype=dtype)

    return xb, wb


def sample_interior(obj, B, replace=False, generator=None,return_idx=False):
    xy_full = obj.xy_full
    N = xy_full.shape[0]
    device = xy_full.device

    if generator is None:
        if replace:
            idx = torch.randint(0, N, (B,), device=device)
        else:
            idx = torch.randperm(N, device=device)[:B]
    else:
        if replace:
            idx = torch.randint(0, N, (B,), device=device, generator=generator)
        else:
            idx = torch.randperm(N, device=device, generator=generator)[:B]

    xyB = xy_full[idx]

    w_full = getattr(obj, "weight_full", None)
    if w_full is None:
        wB = None
    else:
        if torch.is_tensor(w_full) and w_full.numel() == 1:
            wB = w_full          # scalar weight stays scalar
        else:
            wB = w_full[idx]     # pointwise weights subset
    if return_idx:
        return xyB, wB, idx

    return xyB, wB

def train_poisson_with_TR(
    width=64,
    depth=3,
    ngrid=32,
    beta=1e-6,
    delta0=1e-1,
    maxit=200,
    device="cpu",
    seed=None,
):
    # -------------------------
    # model
    # -------------------------
    model = MLPWithFourier(
        mapping_size=8,
        scale=0.5,
        width=width,
        depth=depth,
        activation="relu"
    ).to(device)

    # -------------------------
    # data
    # -------------------------
    xy = make_training_points_grid(ngrid, device=device)
    g = compute_g_from_u_star(xy)

    xb, wb = make_boundary_points(n=ngrid, device=device)
    lam_bc = 100.0

    # weak test space
    modes = [
        (1, 1), (1, 2), (2, 1), (2, 2),
        (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)
    ]
    V, dV = make_sine_tests(xy, modes=modes)

    # -------------------------
    # objective + problem
    # -------------------------
    var = {
        "useEuclidean": False,
        "beta": beta,
    }

    obj_smooth = PoissonWeakPINNObjective(
        model=model,
        xy=xy,
        g=g,
        kappa_fn=kappa_xy,
        V=V,
        dV=dV,
        weight=None,
        device=device,
        mu_I=0.0,
        xb=xb,
        wb=wb,
        bc_target=torch.zeros((xb.shape[0], 1), device=device, dtype=xy.dtype),
        lam_bc=lam_bc,
    )

    obj_nonsmooth = L1TorchNorm(var)

    class _ProblemWrap:
        pass

    problem = _ProblemWrap()
    problem.obj_smooth = obj_smooth
    problem.obj_nonsmooth = obj_nonsmooth
    problem.pvector = L2TVPrimal(var)
    problem.dvector = L2TVDual(var)

    # -------------------------
    # initial parameter vector
    # -------------------------
    x0 = vector_from_model(model)

    # -------------------------
    # TR parameters
    # -------------------------
    params = set_default_parameters("SPG2")

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        params["generator"] = generator

    params["delta"] = delta0
    params["maxit"] = maxit
    params["useInexactObj"] = False
    params["useInexactGrad"] = False

    # deterministic full grid
    x_opt, cnt = trustregion(x0, delta0, problem, params)

    # -------------------------
    # load optimum back into model
    # -------------------------
    load_vector_into_model(x_opt, model)

    return model, x_opt, cnt, problem

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

#derivative check
@torch.no_grad()
def directional_fd_value_model(obj, theta, v, eps):
    th_p = theta.copy()
    th_m = theta.copy()
    th_p.axpy(eps, v)
    th_m.axpy(-eps, v)
    Jp,_ = obj.value_model(th_p)
    Jm,_ = obj.value_model(th_m)
    return (Jp - Jm) / (2.0 * eps)

@torch.no_grad()
def directional_fd_value_true(obj, theta, v, eps):
    th_p = theta.clone()
    th_m = theta.clone()
    th_p.axpy(eps, v)
    th_m.axpy(-eps, v)
    Jp,_ = obj.value(th_p)
    Jm,_ = obj.value(th_m)
    return (Jp - Jm) / (2.0 * eps)

def directional_grad_dot(obj, theta, v):
    g,_ = obj.gradient(theta)
    return g.dot(v)

def grad_check(obj, theta, ntests = 10, eps_list=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    torch.set_default_dtype(torch.float64)
    obj.update(theta, "accept")
    mode = bool(getattr(obj,"_smooth_mode",False))
    print(f"[grad_check] locked smooth_mode = {mode}")
    use_model_fd = mode and hasattr(obj,"value_model")
    for t in range(ntests):
        v = theta.randn_like()
        v.normalize_()
        gTv = directional_grad_dot(obj, theta, v)
        print(f"\nTest {t}: g^T v = {gTv:+.6e}")
        for eps in eps_list:
            if use_model_fd:
                fd = directional_fd_value_model(obj, theta, v, eps)
            else:
                fd = directional_fd_value_true(obj, theta, v, eps)
            relerr = abs(fd - gTv) / max(1.0, abs(fd), abs(gTv))
            print(f" eps={eps: >8.1e} FD={fd:+.6e} relerr={relerr:.3e}")
@torch.no_grad()
def directional_fd_grad_locked(obj, theta, v, eps):
    th_p = theta.copy()
    th_m = theta.copy()
    th_p.axpy(+eps, v)
    th_m.axpy(-eps, v)

    obj.update(th_p, "lock")
    gp, _ = obj.gradient(th_p)

    obj.update(th_m, "lock")
    gm, _ = obj.gradient(th_m)

    out = gp.copy()
    out.axpy(-1.0, gm)
    out.scal(1.0 / (2.0 * eps))

    obj.update(theta, "lock")
    return out


@torch.no_grad()
def hv_check_weak_pinn(obj, theta, ntests=5, eps_list=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4), gradTol=1e-12):
    obj.update(theta, "lock")

    for t in range(ntests):
        v = theta.randn_like()
        v.normalize_()

        Hv, _ = obj.hessVec(v, theta, gradTol=gradTol)
        print(f"\nTest {t}: ||Hv|| = {Hv.norm():.6e}")

        for eps in eps_list:
            fdHv = directional_fd_grad_locked(obj, theta, v, eps)
            diff = fdHv.copy()
            diff.axpy(-1.0, Hv)
            relerr = diff.norm() / max(1.0, fdHv.norm(), Hv.norm())
            print(f" eps={eps:>8.1e} ||FD-Hv||/scale={relerr:.3e}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.autograd.set_detect_anomaly(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # setup
    # -------------------------
    width = 64
    depth = 3
    ngrid = 32
    beta = 1e-8
    delta0 = 1e-1
    maxit = 200
    lam_bc = 100.0

    xy = make_training_points_grid(ngrid, device=device)
    g = compute_g_from_u_star(xy)
    xb, wb = make_boundary_points(n=ngrid, device=device)
    
    mmax = 10
    nmax = 10
    modes = [(i, j) for i in range(1, mmax + 1) for j in range(1, nmax + 1)]

    #modes = [
    #    (1, 1), (1, 2), (2, 1), (2, 2),
    #    (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)
    #]
    V, dV = make_sine_tests(xy, modes=modes)

    # -------------------------
    # pre-training diagnostics
    # -------------------------
    model0 = MLPWithFourier(
        mapping_size=8,
        scale=0.5,
        width=width,
        depth=depth,
        activation="relu"
    ).to(device)

    obj0 = PoissonWeakPINNObjective(
        model=model0,
        xy=xy,
        g=g,
        kappa_fn=kappa_xy,
        V=V,
        dV=dV,
        weight=None,
        device=device,
        mu_I=0.0,
        xb=xb,
        wb=wb,
        bc_target=torch.zeros((xb.shape[0], 1), device=device, dtype=xy.dtype),
        lam_bc=lam_bc,
    )

    x0 = vector_from_model(model0)
    obj0.update(x0, "accept")

    print("\n==== GRAD CHECK at x0 ====")
    grad_check(obj0, x0, ntests=5)

    print("\n==== HV CHECK at x0 ====")
    hv_check_weak_pinn(obj0, x0, ntests=3)

    # -------------------------
    # training
    # -------------------------
    model, x_opt, cnt, problem = train_poisson_with_TR(
        width=width,
        depth=depth,
        ngrid=ngrid,
        beta=beta,
        delta0=delta0,
        maxit=maxit,
        device=device,
        seed=None,
    )

    # -------------------------
    # post-training diagnostics
    # -------------------------
    obj_final = PoissonWeakPINNObjective(
        model=model,
        xy=xy,
        g=g,
        kappa_fn=kappa_xy,
        V=V,
        dV=dV,
        weight=None,
        device=device,
        mu_I=0.0,
        xb=xb,
        wb=wb,
        bc_target=torch.zeros((xb.shape[0], 1), device=device, dtype=xy.dtype),
        lam_bc=lam_bc,
    )

    x_final = vector_from_model(model)
    obj_final.update(x_final, "accept")

    final_rel_l2 = obj_final.relative_L2_error(x_final)
    final_bd_l2 = boundary_L2(model, xb, device=device)

    print(f"\nFinal relative L2 error: {final_rel_l2:.6e}")
    print(f"Final boundary L2:       {final_bd_l2:.6e}")

    if hasattr(model, "kink_score"):
        try:
            ks = model.kink_score(xy, q=0.01)
            print(f"Final kink_score(q=0.01): {ks:.6e}")
        except Exception as e:
            print(f"Could not compute kink_score: {e}")

    print("\n==== GRAD CHECK at x_opt ====")
    grad_check(obj_final, x_final, ntests=5)

    print("\n==== HV CHECK at x_opt ====")
    hv_check_weak_pinn(obj_final, x_final, ntests=3)

    # -------------------------
    # plots
    # -------------------------
    plot_solution_and_error(model, n=121, device=device)
    plot_tr_history(cnt)
