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
    params['gtol']    = 5e-2
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
    params.setdefault('kink_tau', 1e-3)
    params.setdefault('kink_hyst', 5.0)
    # extra gate: require softplus slope ~= ReLU slope
    params.setdefault('grad_match_tol', 0.45)
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





class PoissonCompositeObjective:
    """
    Hybrid-consistent Poisson composite objective (drop-in replacement).

    Goal:
      - Use TRUE ReLU model (value/grad/JVP/VJP/HessVec) when away from kinks.
      - Switch to SMOOTH surrogate consistently (value/grad/JVP/VJP/HessVec)
        when near kink (some preactivations close to 0), with hysteresis.

    This fixes the main inconsistency in your current class:
      value/gradient use ReLU while hessVec uses smooth J/J^T.
    It also fixes a bug in hessVec_d boundary part (missing Jub^T when wb is None)
    and removes mixing of smooth/non-smooth pieces inside one HessVec.

    Requirements:
      - Your model.forward must accept smooth=bool (you already have that).
      - Optional but recommended: model.min_abs_preact(x) for kink detection.
        If absent, it will always stay in TRUE mode (no smoothing).
    """

    def __init__(self, model, xy, g, kappa_fn, weight=None, device="cpu", mu_I=0.0,
                 xb=None, wb=None, bc_target=None, lam_bc=0.0,
                 kink_tau=1e-3, kink_hyst=5.0,grad_match_tol=0.49, grad_match_q=0.90,_smooth_mode=False):
        self.model = model.to(device)
        self.xy = xy.to(device)
        self.g = g.to(device)
        self.kappa_fn = kappa_fn
        self.device = device
        self.xb = xb.to(device) if xb is not None else None
        self.wb = wb.to(device) if wb is not None else None
        self.bc_target = bc_target.to(device) if bc_target is not None else None
        self.lam_bc = float(lam_bc)

        # hybrid switch params
        self.kink_tau = float(kink_tau)
        self.kink_hyst = float(kink_hyst)
        self._smooth_mode = _smooth_mode  # persistent across calls within an iteration (set in update)
        self._force_true = False
        self.grad_match_tol = float(grad_match_tol)
        self.grad_match_q = float(grad_match_q)
        
        # quadrature weights (interior)
        if weight is None:
            self.weight = None
        else:
            w = weight
            if not torch.is_tensor(w):
                w = torch.tensor(w, dtype=self.xy.dtype, device=device)
            self.weight = w.to(device)

        self.mu_I = float(mu_I)
        
        # store full data for later batching
        self.xy_full = xy.detach().clone()
        self.g_full = g.detach().clone()
        self.weight_full = self.weight


        self._last_theta = None

    def set_mu_I(self, mu_I: float):
        self.mu_I = float(mu_I)
    
    def force_true_mode(self):
        self._smooth_mode = False

    # ---------------- update (called by TR framework) ----------------
    def update(self, theta, flag: str):
        # lock mode at init/accept (recommended) so model pieces stay consistent
        self._set_parameters(theta)
        if flag in ("init", "accept", "lock", "model", "trial"):
            self.decide_smooth_mode()
        self._last_theta = None

    # ---------------- helpers ----------------
    @torch.no_grad()
    def _set_parameters(self, theta):
        name_to_param = dict(self.model.named_parameters())
        for k, v in theta.td.items():
            name_to_param[k].copy_(v)

    def _pack_f(self, grad_u: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.cat([grad_u.reshape(-1), u.reshape(-1)], dim=0)

    def _unpack_f(self, z: torch.Tensor):
        N = self.xy.shape[0]
        d = self.xy.shape[1]
        grad_flat = z[:N * d]
        u_flat = z[N * d:]
        grad_u = grad_flat.reshape(N, d)
        u = u_flat.reshape(N, 1)
        return grad_u, u

    @torch.no_grad()
    def decide_smooth_mode(self) -> bool:
        if getattr(self,"_force_true", False):
            self._smooth_mode = False
            return self._smooth_mode
        xy_check = getattr(self, "xy_full", self.xy)
        #kink proximity score
        
        if hasattr(self.model, "kink_score"):
            
            kink_vals = [float(self.model.kink_score(xy_check, q=0.01))]
            if self.xb is not None:
                kink_vals.append(float(self.model.kink_score(self.xb, q=0.01)))
            kink_measure = min(kink_vals)
        elif hasattr(self.model, "min_abs_preact"):
            
            kink_vals = [float(self.model.min_abs_preact(xy_check))]
            if self.xb is not None:
                kink_vals.append(float(self.model.min_abs_preact(self.xb)))
            kink_measure = min(kink_vals)
        else:
            self._smooth_mode = False
            return self._smooth_mode
        
        #gradient match score
        if hasattr(self.model,"softplus_relu_grad_gap"):
            grad_gap_vals = [float(self.model.softplus_relu_grad_gap(
                xy_check, q=self.grad_match_q
            ))]
            if self.xb is not None:
                grad_gap_vals.append(float(self.model.softplus_relu_grad_gap(
                    self.xb, q=self.grad_match_q
                )))
            grad_gap = max(grad_gap_vals)
        else:
            grad_gap = 0.0
            
        kink_close = (kink_measure < self.kink_tau)
        grad_match = (grad_gap < self.grad_match_tol)
        
                

        if not self._smooth_mode:
            if kink_close and grad_match:
                self._smooth_mode = True
        else:
            if (kink_measure > self.kink_hyst * self.kink_tau) or (grad_gap > self.grad_match_tol):
                self._smooth_mode = False
        return self._smooth_mode
    
    # ---------------- stochastic sampling ----------------
    def snapshot_batch(self):
        return (self.xy, self.weight)
    
    def restore_batch(self, snap):
        self.xy, self.weight = snap
        
    def set_batch(self, xy, g=None, weight=None):
        if xy is not None:
            self.xy = xy
        if g is not None:
            self.g = g
        if weight is not None:
            self.weight = weight
            
        #print("batch:", self.xy.shape,self.g.shape,None if self.weight is None else self.weight.shape)
            
        self._last_theta = None
        
    def value_true_full(self, theta):
        snap = self.snapshot_batch()
        self.set_batch(self.xy_full, self.g_full, self.weight_full)
        self.update(theta, "lock")
        val,_ = self.value(theta, 1e-12)
        self.restore_batch(snap)
        return val
    

    # ---------------- model forward wrappers ----------------
    def _model_forward(self, x, smooth: bool):
        if "smooth" in self.model.forward.__code__.co_varnames:
            return self.model(x, smooth=bool(smooth))
        else:
            return self.model(x)

    def u_boundary_current(self):
        if self.xb is None:
            return None
        return self._model_forward(self.xb, self._smooth_mode)

    def _u_with_params(self, params, xy, smooth: bool):
        buffers = dict(self.model.named_buffers())
        if "smooth" in self.model.forward.__code__.co_varnames:
            return functional_call(self.model, (params, buffers), (xy,), kwargs={"smooth": bool(smooth)})
        else:
            return functional_call(self.model, (params, buffers), (xy,))

    # ---------------- f(theta) and h(z) ----------------
    def f(self, theta):
        self._set_parameters(theta)
        xy = self.xy.detach().clone().requires_grad_(True)

        # TRUE objective always: smooth=False
        if "smooth" in self.model.forward.__code__.co_varnames:
            u = self.model(xy, smooth=False)
        else:
            u = self.model(xy)

        grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        return self._pack_f(grad_u, u)

    def h(self, z: torch.Tensor) -> torch.Tensor:
        grad_u, u = self._unpack_f(z)

        kap = self.kappa_fn(self.xy)
        if kap.ndim == 1:
            kap = kap.reshape(-1, 1)

        integrand = 0.5 * kap * (grad_u**2).sum(dim=1, keepdim=True) - self.g * u

        if self.weight is None:
            return integrand.mean()
        else:
            w = self.weight
            if w.numel() == 1:
                return w * integrand.sum()
            if w.ndim == 1:
                w = w.reshape(-1, 1)
            return (w * integrand).sum()

    # ---------------- objective value ----------------
    def value(self, theta, ftol=1e-12):
        with torch.enable_grad():
            self._set_parameters(theta)

            xy = self.xy.detach().clone().requires_grad_(True)

            # TRUE objective always: smooth=False
            if "smooth" in self.model.forward.__code__.co_varnames:
                u = self.model(xy, smooth=False)
            else:
                u = self.model(xy)

            grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
            z = self._pack_f(grad_u, u)
            val = self.h(z)

            # boundary term also TRUE objective (smooth=False)
            if (self.xb is not None) and (self.lam_bc > 0.0):
                if "smooth" in self.model.forward.__code__.co_varnames:
                    ub = self.model(self.xb, smooth=False)
                else:
                    ub = self.model(self.xb)

                target = 0.0 if self.bc_target is None else self.bc_target
                diff = ub - target
                if self.wb is None:
                    bc_term = 0.5 * self.lam_bc * (diff**2).mean()
                else:
                    w = self.wb.reshape(-1, 1) if self.wb.ndim == 1 else self.wb
                    bc_term = 0.5 * self.lam_bc * (w * (diff**2)).sum()
                val = val + bc_term

        return float(val.detach().cpu().item()), 0.0
    
    def value_model(self, theta, ftol=1e-12):
        """
        Model-consistent value used ONLY for predicted reduction / TR subproblem baseline.

        - If _smooth_mode=False: equals true value (ReLU).
        - If _smooth_mode=True : evaluates the SAME functional but with smooth surrogate forward
          so that (val_model, grad, hessVec) are consistent.

        Acceptance should STILL use self.value(...) (true ReLU).
        """
        with torch.enable_grad():
            self._set_parameters(theta)
            xy = self.xy.detach().clone().requires_grad_(True)

            # MODEL value uses current derivative mode
            if "smooth" in self.model.forward.__code__.co_varnames:
                u = self.model(xy, smooth=bool(self._smooth_mode))
            else:
                u = self.model(xy)

            grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
            z = self._pack_f(grad_u, u)
            val = self.h(z)

            # boundary term in the SAME mode
            if (self.xb is not None) and (self.lam_bc > 0.0):
                if "smooth" in self.model.forward.__code__.co_varnames:
                    ub = self.model(self.xb, smooth=bool(self._smooth_mode))
                else:
                    ub = self.model(self.xb)

                target = 0.0 if self.bc_target is None else self.bc_target
                diff = ub - target
                if self.wb is None:
                    bc_term = 0.5 * self.lam_bc * (diff**2).mean()
                else:
                    w = self.wb.reshape(-1, 1) if self.wb.ndim == 1 else self.wb
                    bc_term = 0.5 * self.lam_bc * (w * (diff**2)).sum()
                val = val + bc_term

        return float(val.detach().cpu().item()), 0.0

    # ---------------- gradient wrt theta ----------------
    def gradient(self, theta, gtol=1e-12):
        self._set_parameters(theta)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            xy = self.xy.detach().clone().requires_grad_(True)

            # DERIVATIVE model: switch by _smooth_mode
            if "smooth" in self.model.forward.__code__.co_varnames:
                u = self.model(xy, smooth=bool(self._smooth_mode))
            else:
                u = self.model(xy)

            grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
            val = self.h(self._pack_f(grad_u, u))

            # boundary derivative must follow SAME mode as gradient model
            if (self.xb is not None) and (self.lam_bc > 0.0):
                if "smooth" in self.model.forward.__code__.co_varnames:
                    ub = self.model(self.xb, smooth=bool(self._smooth_mode))
                else:
                    ub = self.model(self.xb)

                target = 0.0 if self.bc_target is None else self.bc_target
                diff = ub - target
                if self.wb is None:
                    bc_term = 0.5 * self.lam_bc * (diff**2).mean()
                else:
                    w = self.wb.reshape(-1, 1) if self.wb.ndim == 1 else self.wb
                    bc_term = 0.5 * self.lam_bc * (w * (diff**2)).sum()
                val = val + bc_term

            val.backward()

        grad_td = OrderedDict((name, p.grad.detach().clone()) for name, p in self.model.named_parameters())
        return TorchDictVector(grad_td), 0.0
    # =========================================================
    # Functorch-safe f_of_params for JVP/VJP (hybrid-consistent)
    # =========================================================
    def _f_of_params_functorch_hybrid(self, params):
        xy = self.xy
        smooth = self._smooth_mode

        # u(xy)
        u = self._u_with_params(params, xy, smooth)

        # grad_x u(xy) via torch.func.grad on the whole batch
        grad_u = grad(lambda X: self._u_with_params(params, X, smooth).sum())(xy)

        return self._pack_f(grad_u, u)

    def _ub_of_params_functorch_hybrid(self, params):
        if self.xb is None:
            return None
        smooth = self._smooth_mode
        ub = self._u_with_params(params, self.xb, smooth)
        return ub.reshape(-1)

    # ---------------- JVP/VJP (single, consistent) ----------------
    def apply_Jf_functorch(self, theta, s):
        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}
        tang = {k: s.td[k] for k, _ in self.model.named_parameters()}
        _, Jd = jvp(self._f_of_params_functorch_hybrid, (params0,), (tang,))
        return Jd

    def apply_JfT_functorch(self, theta, cotangent_z):
        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}
        _, pullback = vjp(self._f_of_params_functorch_hybrid, params0)
        grads = pullback(cotangent_z)[0]
        hv_td = OrderedDict((name, grads[name].detach().clone()) for name, _ in self.model.named_parameters())
        return TorchDictVector(hv_td)

    def apply_Jub_functorch(self, theta, s):
        if self.xb is None:
            return None
        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}
        tang = {k: s.td[k] for k, _ in self.model.named_parameters()}
        _, Jubv = jvp(self._ub_of_params_functorch_hybrid, (params0,), (tang,))
        return Jubv

    def apply_JubT_functorch(self, theta, cotangent_ub):
        if self.xb is None:
            return None
        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}
        _, pullback = vjp(self._ub_of_params_functorch_hybrid, params0)
        grads = pullback(cotangent_ub)[0]
        hv_td = OrderedDict((name, grads[name].detach().clone()) for name, _ in self.model.named_parameters())
        return TorchDictVector(hv_td)

    # ---------------- predicted reduction ----------------
    def predicted_reduction(self, theta, s):
        with torch.enable_grad():
            z = self.f(theta)
            Jd = self.apply_Jf_functorch(theta, s)
            pred = self.h(z) - self.h(z + Jd)
        return float(pred.detach().cpu().item())

    # ---------------- Hessian-vector (Gauss-Newton) ----------------
    def hessVec(self, v, theta, gradTol=1e-12):
        """
        Consistent GN curvature using the SAME mode as value/gradient:
          interior: J^T [scale * Jv_grad ; 0]
          boundary: lam * Jub^T * W * Jubv   (or mean if wb is None)
          + mu_I * v
        """
        with torch.enable_grad():
            # ---- interior GN ----
            
            Jv = self.apply_Jf_functorch(theta, v)
            

            N = self.xy.shape[0]
            d = self.xy.shape[1]
            kap = self.kappa_fn(self.xy)
            if kap.ndim == 1:
                kap = kap.reshape(-1, 1)

            if self.weight is None:
                scale = (1.0 / float(N)) * kap
            else:
                if self.weight.numel() == 1:
                    scale = self.weight * kap
                else:
                    ww = self.weight.reshape(-1, 1) if self.weight.ndim == 1 else self.weight
                    scale = ww * kap

            Jv_grad = Jv[:N * d].reshape(N, d)
            Jv_u = Jv[N * d:]

            Hz_grad = scale * Jv_grad
            Hz_u = torch.zeros_like(Jv_u)
            Hz = torch.cat([Hz_grad.reshape(-1), Hz_u.reshape(-1)], dim=0)

            hv = self.apply_JfT_functorch(theta, Hz)

            # ---- boundary GN ----
            if (self.xb is not None) and (self.lam_bc > 0.0):
                
                Jubv = self.apply_Jub_functorch(theta, v)
                    
                if self.wb is None:
                    Nb = Jubv.numel()
                    cot = (self.lam_bc / float(Nb)) * Jubv
                else:
                    w = self.wb.reshape(-1)
                    cot = self.lam_bc * (w * Jubv)
                
                hv_bc = self.apply_JubT_functorch(theta, cot)
                
                hv = hv + hv_bc

        if self.mu_I != 0.0:
            hv = hv + (self.mu_I * v)
        return hv, 0.0

    # ---------------- Optional: keep your GN smooth gradient (still useful) ----------------
    def gradient_gn_smooth(self, theta):
        """
        This computes GN gradient for the *smooth* surrogate only.
        Kept for compatibility with your checks; not used by TR unless you call it.
        """
        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}
        z = self._f_of_params_functorch_hybrid(params0)  # uses current mode (might be smooth or true)

        N, d = self.xy.shape
        grad_u, u = self._unpack_f(z)
        kap = self.kappa_fn(self.xy)
        if kap.ndim == 1:
            kap = kap.reshape(-1, 1)

        if self.weight is None:
            scale = (1.0 / float(N)) * kap
            dh_du = (-(1.0 / float(N)) * self.g).reshape(-1)
        else:
            if self.weight.numel() == 1:
                scale = self.weight * kap
                dh_du = (-(self.weight) * self.g).reshape(-1)
            else:
                ww = self.weight.reshape(-1, 1) if self.weight.ndim == 1 else self.weight
                scale = ww * kap
                dh_du = (-(ww) * self.g).reshape(-1)

        dh_dgrad = (scale * grad_u).reshape(-1)
        cotangent = torch.cat([dh_dgrad, dh_du], dim=0)

        g = self.apply_JfT_functorch(theta, cotangent)

        if (self.xb is not None) and (self.lam_bc > 0.0):
            ub = self._ub_of_params_functorch_hybrid(params0)
            target = 0.0 if self.bc_target is None else self.bc_target.reshape(-1)
            r = (ub - target)
            if self.wb is None:
                Nb = r.numel()
                cot = (self.lam_bc / float(Nb)) * r
            else:
                w = self.wb.reshape(-1) if self.wb.ndim == 2 else self.wb.reshape(-1)
                cot = self.lam_bc * (w * r)
            g_bc = self.apply_JubT_functorch(theta, cot)
            g = g + g_bc

        return g

    # ---------------- relative L2 error diagnostic ----------------
    def relative_L2_error(self, theta):
        self._set_parameters(theta)
        with torch.no_grad():
            xy = self.xy
            u_pred = self._model_forward(xy, smooth=False)
            u_true = u_star(xy)
            N = xy.shape[0]
            n_side = int(N ** 0.5)
            h = 1.0 / (n_side - 1)
            weight = h * h
            err_sq = weight * torch.sum((u_pred - u_true) ** 2)
            true_sq = weight * torch.sum(u_true ** 2)
            rel_L2 = torch.sqrt(err_sq / true_sq)
        return rel_L2.item()
    

@torch.no_grad()
def boundary_L2(model, xb, device="cpu"):
    ub = model(xb.to(device))
    return torch.sqrt(torch.mean(ub**2)).item() 

    

def full_gnorm(theta, problem, params):
    obj = problem.obj_smooth

    # switch to FULL dataset
    if hasattr(obj, "set_batch"):
        obj.set_batch(obj.xy_full, obj.g_full, obj.weight_full)
    else:
        obj.xy = obj.xy_full
        obj.g = obj.g_full
        obj.weight = obj.weight_full

    obj.update(theta, "accept")

    # true gradient
    g, _ = obj.gradient(theta)

    # composite proximal gradient norm
    d = problem.dvector.dual(g)
    xprox = problem.obj_nonsmooth.prox(
        theta - params['ocScale'] * d,
        params['ocScale']
    )

    pg = xprox.copy()
    pg.axpy(-1.0, theta)

    return problem.pvector.norm(pg) / params['ocScale']


@torch.no_grad()
def full_pgnorm(x, problem, params, t=None):
    obj = problem.obj_smooth

    # restore FULL batch
    if hasattr(obj, "set_batch"):
        obj.set_batch(obj.xy_full, obj.g_full, obj.weight_full)
    else:
        obj.xy, obj.g, obj.weight = obj.xy_full, obj.g_full, obj.weight_full

    # lock smooth_mode consistently at this x
    obj.update(x, "accept")

    # full smooth gradient
    g, _ = obj.gradient(x)

    # prox-gradient mapping norm (composite stationarity)
    if t is None:
        t = params.get("t", 1.0)

    x_tmp = x.copy()
    x_tmp.axpy(-t, g)                       # x - t*g
    x_prox = problem.obj_nonsmooth.prox(x_tmp, t)  # prox_{t phi}(x - t*g)

    G = x.copy()
    G.axpy(-1.0, x_prox)                    # x - prox(...)
    G.scal(1.0 / t)

    return float(G.norm())

# -------------------------
# 2) ReLU network
# -------------------------   


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


class PoissonNet(nn.Module):
    """
    u_theta(x) = b(x) * com_theta(x)
    so Dirichlet BC u=0 is satisfied automatically.
    """
    def __init__(self, in_dim=2, width=64, depth=3, activation="relu"):
        super().__init__()
        self.com = MLP(in_dim=in_dim, width=width, depth=depth, out_dim=1, activation=activation)
        self.smooth_beta = 50.0

    def forward(self, xy, smooth = False):
        if not smooth:
            return b_factor(xy) * self.com(xy) 
        else:
            return b_factor(xy) *self.forward_smooth(xy) 
    
    def forward_smooth(self, xy):
        h = xy
        beta = self.smooth_beta
        for layer in self.com.net:
            if isinstance(layer, nn.ReLU):
                h = F.softplus(h, beta = beta) / beta
            else:
                h = layer(h)
        return h
        
    
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


def trustregion_dropin_stochastic(x0, Deltai, problem, params):
    """
    Drop-in TR loop with a robust FORCE_TRUE fallback.

    Key rules (enforced here):
      1) Build the step on a MINI-BATCH (model) normally.
      2) Accept/reject using FULL objective.
      3) If a step is rejected, enter FORCE_TRUE mode:
           - temporarily switch to FULL batch for model building
           - use TRUE value (obj.value) as the model anchor (val_model) in the subproblem
           - exit FORCE_TRUE after a few consecutive accepts
      4) Always clamp step to trust region radius (s-norm <= delta).

    Requirements (same as your code):
      - obj = problem.obj_smooth
      - obj.value(theta, tol)            : TRUE objective (full, nonsmooth excluded)
      - obj.value_model(theta, tol)      : stochastic/model objective on current batch
      - obj.hessVec(v, theta)            : Hessian-vector product (uses obj._smooth_mode)
      - compute_gradient(x, problem, params, cnt) -> (grad, dgrad, gnorm, cnt)
      - trustregion_step_SPG2(...) -> (s, snorm, pRed, phinew_model, iflag, iter_count, cnt, params)
      - full_pgnorm(x, problem, params)
      - sample_interior(obj, batch_size, ...) (your sampler) OR you can replace with your own
      - problem.obj_nonsmooth.value(theta) and prox(...) exist

    NOTE:
      This function never assumes xy_m/g_m/w_m. It uses obj.xy_full/g_full/weight_full.
    """

    start_time = time.time()

    # ---------------- defaults ----------------
    params = dict(params)  # local copy
    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 500)
    params.setdefault('gtol', 5e-2)
    params.setdefault('stol_abs', 1e-9)
    params.setdefault('delta_stop', 1e-7)
    params.setdefault('atol', 1e-4)
    params.setdefault('rtol', 1e-2)
    params.setdefault('spexp', 2)
    params.setdefault('eta1', 0.05)
    params.setdefault('eta2', 0.5)
    params.setdefault('gamma1', 0.25)
    params.setdefault('gamma2', 1.5)
    params.setdefault('delta', Deltai)
    params.setdefault('deltamin', 1e-16)
    params.setdefault('deltamax', 1.0)
    params.setdefault('reltol', False)
    params.setdefault('stag_window', 10)
    params.setdefault('ftol_rel', 1e-6)
    params.setdefault('max_reject', 7)
    params.setdefault("nonmono_M", 10)
    params.setdefault("full_grad_every", 10)

    # stochastic controls
    params.setdefault("batch_size", None)      # e.g. 512; None => full-batch
    params.setdefault("sample_every", 1)       # resample every k iters

    # force-true controls
    params.setdefault("force_true_accepts_to_exit", 5)  # consecutive accepts to exit FORCE_TRUE
    params.setdefault("force_true_fullbatch", False)     # FORCE_TRUE uses full batch for model build
    
    # new termination conditions
    params.setdefault("pred_abs_tol", 1.5e-8)   # absolute floor for predicted decrease
    params.setdefault("pred_rel_tol", 1e-9)   # relative floor w.r.t. |val_model+phi|
    params.setdefault("pred_small_max", 5)     # how many consecutive "tiny" before stop
    params.setdefault("pred_check_every", 1)   # check each iter 

    # optional: regularization in smooth mode
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
        # --- add these to match compute_gradient() ---
        'valerr': [],
        'valtol': [],
        'graderr': [],
        'gradtol': [],
     }

    obj = problem.obj_smooth

    # ---------------- init x ----------------
    if params['initProx']:
        x = problem.obj_nonsmooth.prox(x0, 1.0)
        cnt['nprox'] += 1
    else:
        x = x0.copy()

    # ---------------- store FULL data once ----------------
    if not hasattr(obj, "xy_full"):
        obj.xy_full = obj.xy
        obj.g_full = obj.g
        obj.weight_full = getattr(obj, "weight", None)

    # ---------------- helper: set batch ----------------
    def _assign_batch(xy_b, g_b, w_b):
        if hasattr(obj, "set_batch"):
            obj.set_batch(xy_b, g_b, w_b)
        else:
            obj.xy = xy_b
            obj.g = g_b
            obj.weight = w_b

    def _set_batch_from_full(batch_size, replace=False, generator=None):
        N = obj.xy_full.shape[0]
        if (batch_size is None) or (batch_size >= N):
            xy_b = obj.xy_full
            g_b  = obj.g_full
            w_b  = obj.weight_full
        else:
            # Your sampler should return xy_b, w_b, idx (you already have sample_interior)
            xy_b, w_b, idx = sample_interior(obj, batch_size, replace=replace,
                                             generator=generator, return_idx=True)
            g_b = obj.g_full[idx]
        _assign_batch(xy_b, g_b, w_b)

        # lock smooth_mode for this batch at current x
        obj.update(x, "lock")

    # ---------------- initial batch (for model) ----------------
    _set_batch_from_full(params["batch_size"], replace=False, generator=params.get('generator'))

    # ---------------- initial TRUE eval (full) ----------------
    _assign_batch(obj.xy_full, obj.g_full, obj.weight_full)
    obj.update(x, "accept")

    val_true, _ = obj.value(x, 1e-12); cnt['nobj1'] += 1
    phi = problem.obj_nonsmooth.value(x); cnt['nobj2'] += 1

    # gradient for printing (can be batch-grad; true check later)
    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
    
    

    Facc = [val_true + phi]
    Fhist = deque(maxlen=params["nonmono_M"])
    Fhist.append(val_true + phi)

    print(f"TR method using {params.get('spsolver','SPG2')} Subproblem Solver (stochastic model)")
    print("  iter    value    gnorm    del    snorm    nobjs    ngrad    nobjn    nprox     iterSP    flagSP")
    print(f"{0:4d} {0:4d} {val_true+phi:8.6e} {params['delta']:8.6e}  ---      "
          f"{cnt['nobj1']:6d}      {cnt['ngrad']:6d}      {cnt['nobj2']:6d}      {cnt['nprox']:6d} --- ---")

    # history
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

    rej_count = 0
    force_true_accepts = 0
    small_pred_count = 0
    if not hasattr(obj, "_force_true"):
        obj._force_true = False
        
    

    # ==========================
    # main loop
    # ==========================
    for i in range(1, params['maxit'] + 1):

        # (A) choose batch for model build
        # If FORCE_TRUE, optionally force full batch for model build
        bs = params["batch_size"]
        if getattr(obj, "_force_true", False) and params["force_true_fullbatch"]:
            bs = None

        if (i == 1) or (params["sample_every"] and (i % params["sample_every"] == 0)) or getattr(obj, "_force_true", False):
            _set_batch_from_full(bs, replace=False, generator=params.get('generator'))

        # (B) regularize only in smooth mode (optional)
        if getattr(obj, "_smooth_mode", False):
            if hasattr(obj, "set_mu_I"):
                obj.set_mu_I(params["mu_smooth"])
        else:
            if hasattr(obj, "set_mu_I"):
                obj.set_mu_I(0.0)
                
        def _set_full_batch_temporarily():
            snap = obj.snapshot_batch()
            _assign_batch(obj.xy_full, obj.g_full, obj.weight_full)
            return snap

        def _restore_batch(snap):
            obj.restore_batch(snap)
            
        snap_force = None
        if getattr(obj, "_force_true", False):
            # ensure model anchor + grad + Hv all see the SAME data
            snap_force = obj.snapshot_batch()
            _assign_batch(obj.xy_full, obj.g_full, obj.weight_full)
            obj.update(x, "lock")  # lock any mode decisions (if any)

        # (C) model anchor value for subproblem
        # IMPORTANT: if FORCE_TRUE => use TRUE objective as model anchor
        if getattr(obj, "_force_true", False):
            val_model, _ = obj.value(x, 1e-12)
        else:
            val_model, _ = obj.value_model(x, 1e-12)
        cnt['nobj1'] += 1

        # (D) gradient for subproblem model (batch-consistent)
        grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
        
        # For new termination condition
        if (params["pred_check_every"] and (i % params["pred_check_every"] == 0)):
            sc,snormc, pRed_gcp, phinew_gcp, _, cnt, params = trustregion_gcp2(x, val_model, grad, dgrad, phi, problem, params, cnt)
            floor = max(params["pred_abs_tol"], params["pred_rel_tol"]*max(1.0, abs(val_model+phi)))
            if float(pRed_gcp) <= floor:
                small_pred_count += 1
            else:
                small_pred_count = 0
            if (small_pred_count >= params["pred_small_max"]) and (params["delta"] <= params["delta_stop"]):
                print("Optimization terminated: predicted model decrease is tiny (GCP) and TR radius is small.")
                cnt['iter'] = i
                cnt['timetotal'] = time.time() - start_time
                cnt['iflag'] = 6 
                return x, cnt
        
            
        

        # (E) solve TR subproblem
        params['tolsp'] = min(params['atol'], params['rtol'] * (gnorm ** params['spexp']))
        s, snorm, pRed, phinew_model, iflag, iter_count, cnt, params = trustregion_step_SPG2(
            x, val_model, grad, dgrad, phi, problem, params, cnt
        )
        
        pred_floor = max(params["pred_abs_tol"],
                 params["pred_rel_tol"] * max(1.0, abs(val_model + phi)))

        if float(pRed) <= pred_floor and params["delta"] <= params["delta_stop"]:
            print("Terminate: SPG2 predicted reduction tiny and delta small.")
            cnt['iter'] = i
            cnt['timetotal'] = time.time() - start_time
            cnt['iflag'] = 7
            return x, cnt 
        
        if snap_force is not None:
            obj.restore_batch(snap_force)

        # enforce trust region radius (hard clamp)
        sn = problem.pvector.norm(s)
        if sn > params['delta'] * (1 + 1e-12):
            params['delta'] = max(params['deltamin'], params['gamma1'] * params['delta'])
            rej_count += 1
            continue

        xnew = x + s

        # (F) predicted reduction pRed computed CONSISTENTLY with mode
        
        if params.get("debug_pred", False):
            gs = grad.dot(s)
            Hs, _ = obj.hessVec(s, x)
            sHs = s.dot(Hs)
            m_xnew = val_model + gs + 0.5 * sHs
            pRed_dbg = float((val_model + phi) - (m_xnew + phinew_model))
            print("pred parts dbg:", float(val_model+phi), float(m_xnew+phinew_model),
                  "pRed_sub=", float(pRed), "pRed_dbg=", pRed_dbg)

        # (G) TRUE acceptance test on FULL dataset
        _assign_batch(obj.xy_full, obj.g_full, obj.weight_full)
        obj.update(xnew, "trial")

        valnew_true, _ = obj.value(xnew, 1e-12); cnt['nobj1'] += 1
        phinew_true = problem.obj_nonsmooth.value(xnew); cnt['nobj2'] += 1

        aRed = float((val_true + phi) - (valnew_true + phinew_true))

        # nonmonotone filter
        Fref = max(Fhist)
        accept_nm = (valnew_true + phinew_true) <= (Fref - 1e-12)

        # robust rho / accept logic
        pred_floor = 1e-10 * max(1.0, abs(val_true + phi))
        if pRed <= pred_floor:
            rho = -np.inf
            accept = (aRed >= 0.0) and accept_nm
        else:
            rho = aRed / pRed
            accept = (rho >= params['eta1']) and accept_nm

        # debug
        mode = "FORCE_TRUE" if getattr(obj, "_force_true", False) else "MODEL"
        print(f"DEBUG mode: {mode} smooth_mode={bool(getattr(obj,'_smooth_mode',False))} pRed={pRed}")
        print("debug:", "aRed=", aRed, "pRed=", pRed, "rho=", float(rho),
              "accept_nm=", bool(accept_nm))

        # (H) update TR radius and mode flags
        if not accept:
            # shrink delta (use min(delta, sn) so we don't keep huge delta after a tiny step)
            sn = problem.pvector.norm(s)
            params['delta'] = max(params['deltamin'], params['gamma1'] * min(params['delta'], sn))

            # enter FORCE_TRUE
            obj._force_true = True
            obj._smooth_mode = False
            if hasattr(obj,"set_mu_I"):
                obj.set_mu_I(0.0)
            force_true_accepts = 0
            obj.update(x, 'reject')
            rej_count += 1

        else:
            # accept
            x = xnew
            phi = phinew_true
            val_true = valnew_true

            rej_count = 0
            Facc.append(val_true + phi)
            Fhist.append(val_true + phi)

            # if we were in FORCE_TRUE, count consecutive accepts to exit
            if getattr(obj, "_force_true", False):
                force_true_accepts += 1
                if force_true_accepts >= params["force_true_accepts_to_exit"]:
                    obj._force_true = False
                    force_true_accepts = 0
            else:
                force_true_accepts = 0

            # enlarge if good agreement
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

        # FULL PG check
        if i % params["full_grad_every"] == 0:
            pgnorm_true = full_pgnorm(x, problem, params)
            print("debug: full  PG gnorm =", pgnorm_true)
            if pgnorm_true <= params["gtol"]:
                print("Optimization terminated because full gradient tolerance met")
                cnt['iter'] = i
                cnt['iflag'] = 0
                cnt['timetotal'] = time.time() - start_time
                return x, cnt

        # stopping checks
        stop_grad = (gnorm <= gtol)
        stop_step = (snorm < params["stol_abs"]) and (params["delta"] <= params["delta_stop"])
        stop_stuck = (params["delta"] <= 10 * params["delta_stop"] and rej_count >= params["max_reject"])

        stop_stag = False
        K = params["stag_window"]
        if len(Facc) >= K + 1:
            Fold = Facc[-(K+1)]
            Fnew = Facc[-1]
            rel_change = abs(Fold - Fnew) / max(1.0, abs(Fnew))
            stop_stag = rel_change < params["ftol_rel"]

        stop_maxit = (i >= params["maxit"])

        if stop_grad or stop_step or stop_stag or stop_stuck or stop_maxit:
            if stop_grad:
                flag, reason = 0, "gradient tolerance met (batch)"
            elif stop_step:
                flag, reason = 2, "step small and TR radius collapsed"
            elif stop_stag:
                flag, reason = 3, "objective stagnation"
            elif stop_stuck:
                flag, reason = 4, "trust region stuck (delta small + rejections)"
            else:
                flag, reason = 1, "maximum iterations reached"

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


def trustregion(x0, Deltai, problem, params):
    start_time = time.time()

    # --- defaults (do not use setdefault here if you want these to always apply) ---
    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 500)
    params.setdefault('gtol', 5e-2)
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
    params.setdefault('delta_stop',1e-7)
    params.setdefault('stol_abs', 1e-9)
    params.setdefault('stag_window', 10)
    params.setdefault('ftol_rel',1e-6)
    params.setdefault('max_reject',15)
    
    params.setdefault("batch_size_model", 512)   # for grad/Hv/subproblem model
    params.setdefault("batch_size_acc",   4096)  # for acceptance objective
    params.setdefault("batch_replace",    False) # sample without replacement
    params.setdefault("batch_seed",       None)  # optional reproducibility
    
    def sample_batch(obj, B, replace=False, seed=None):
        N = obj.xy_full.shape[0]  # we'll set xy_full once below
        g = None
        if seed is not None:
            g = torch.Generator(device=obj.xy_full.device)
            g.manual_seed(int(seed))
        idx = torch.randint(0, N, (B,), generator=g, device=obj.xy_full.device) if replace else torch.randperm(N, generator=g, device=obj.xy_full.device)[:B]
        xyB = obj.xy_full[idx]
        wB = None
        if obj.weight_full is not None:
            wB = obj.weight_full[idx]
        return xyB, wB
    

    # --- counters/history ---
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
 
    # --- init x ---
    
    if params['initProx']:
        x = problem.obj_nonsmooth.prox(x0, 1.0)
        cnt['nprox'] += 1
    else:
        x = x0.copy()
    
    #obj = problem.obj_smooth
    #if not hasattr(obj, "xy_full"):
    #    obj.xy_full = obj.xy
    #    obj.weight_full = obj.weight

    problem.obj_smooth.update(x, "init")
    # store full interior set once (so we can swap batches)
    problem.obj_smooth.xy_full = problem.obj_smooth.xy
    problem.obj_smooth.weight_full = problem.obj_smooth.weight

    # --- initial eval ---
    rej_count = 0
    val_true, _ = problem.obj_smooth.value(x, 1e-12)
    cnt['nobj1'] += 1
    if hasattr(problem.obj_smooth, "value_model"):
        val_model, _ = problem.obj_smooth.value_model(x, 1e-12)
        cnt['nobj1'] += 1
    else:
        val_model = val_true
    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
    phi = problem.obj_nonsmooth.value(x)
    cnt['nobj2'] += 1
    Facc = [val_true+phi]
    
    #ADD FOR RELU WORKING WELL
    M = params.get("nonmono_M", 10)
    Fhist = deque(maxlen=M)
    Fhist.append(val_true+phi)
    

    # --- header ---
    print(f"TR method using {params.get('spsolver','SPG2')} Subproblem Solver")
    print("  iter    value    gnorm    del    snorm    nobjs    ngrad    nobjn    nprox     iterSP    flagSP")
    print(f"{0:4d} {0:4d} {val_true+phi:8.6e} {params['delta']:8.6e}  ---      {cnt['nobj1']:6d}      {cnt['ngrad']:6d}      {cnt['nobj2']:6d}      {cnt['nprox']:6d} --- ---")

    # --- store init ---
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

    # --- stopping tolerances ---
    gtol = params['gtol']
    stol = params['stol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm
        stol = params['stol'] * gnorm

    # ==========================
    # main loop
    # ==========================
    for i in range(1, params['maxit'] + 1):

        # solve TR subproblem
        # bigger mu for small delta -> more conservative, more convex model
        #problem.obj_smooth.set_mu(1.0 )#/ max(params['delta'], 1e-6))
        #problem.obj_smooth._set_parameters(x) 
        #problem.obj_smooth.decide_smooth_mode()

        params['tolsp'] = min(params['atol'], params['rtol'] * (gnorm ** params['spexp']))
        if problem.obj_smooth._smooth_mode:
            problem.obj_smooth.set_mu_I(1e-4)  # try 1e-4 to 1e-3
        else:
            problem.obj_smooth.set_mu_I(0.0)
            
        
        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
            x, val_model, grad, dgrad, phi, problem, params, cnt
        )

        # evaluate trial
        xnew = x + s
        #problem.obj_smooth.update(xnew, 'trial')
        #val_true, val_old, cnt = compute_value(xnew, x, val, problem.obj_smooth, pRed, params, cnt)
        valnew_true, _ = problem.obj_smooth.value(xnew, 1e-12)
        cnt['nobj1'] += 1
        phinew_true = problem.obj_nonsmooth.value(xnew)
        cnt['nobj2'] += 1

        
        #print("trial diagnostics:","val=", val, "phi=", phi,"valnew=", valnew, "phi_true=", phinew_true,"pRed=", pRed, "aRed=", aRed)
        #Add for relu
        aRed = (val_true + phi) - (valnew_true + phinew_true)
        
        pRed      = float(pRed)
        rho = -np.inf if pRed <= 0 else float(aRed) / pRed
        Fref = max(Fhist)
        accept_nm = (valnew_true + phinew_true) <= (Fref - 1e-12)

        accept =(rho >= params['eta1']) and accept_nm
        
        print("debug:","aRed=", float(aRed), "pRed=", float(pRed), "rho=", float(rho))#,"rho_ref=", float(rho_ref), "accept=", bool(accept))
        if not accept:
            params['delta'] = max(params['deltamin'],params['gamma1'] * params['delta'])
            problem.obj_smooth.update(x,'reject')
            rej_count += 1
       
        
        else:
        
            #accept
            x = xnew
            phi = phinew_true
            rej_count = 0
            #problem.obj_smooth._set_parameters(x)
            #problem.obj_smooth.decide_smooth_mode()
            problem.obj_smooth.update(x, 'accept')
            val_true, _ = problem.obj_smooth.value(x, 1e-12)
            cnt['nobj1'] += 1
            if hasattr(problem.obj_smooth, "value_model"):
                val_model, _ = problem.obj_smooth.value_model(x,1e-12)
                cnt['nobj1'] += 1
            else:
                val_model = val_true

            grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
            
            Facc.append(val_true+phi)

            Fhist.append(val_true+phi)
            relL2 = problem.obj_smooth.relative_L2_error(x)
            print("relative L2 error =", relL2)
            print("boundary L2 =", boundary_L2(problem.obj_smooth.model, problem.obj_smooth.xb, device))
            
            if rho > params['eta2']:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])

        
        if i % params['outFreq'] == 0:
            print(f"{i:4d}    {val_true + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      "
                  f"{cnt['nobj1']:6d}     {cnt['ngrad']:6d}       {cnt['nobj2']:6d}     {cnt['nprox']:6d}      "
                  f"{iter_count:4d}        {iflag:1d}")

        # store
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

        # stopping 
        delta_stop = params["delta_stop"]
        stol_abs   = params["stol_abs"]
        K          = params["stag_window"]
        ftol_rel   = params["ftol_rel"]
        max_reject = params["max_reject"]

        # ---- stopping tests ----
        stop_grad = (gnorm <= gtol)

        # small step ONLY meaningful if TR radius collapsed
        stop_step = (snorm < stol_abs) and (params["delta"] <= delta_stop)

        # trust-region stuck (many rejections)
        stop_stuck = (params["delta"] <= 10*delta_stop and rej_count >= max_reject)

        # objective stagnation
        stop_stag = False
        if len(Facc) >= K + 1:
            Fold = Facc[-(K+1)]
            Fnew = Facc[-1]
            rel_change = abs(Fold - Fnew) / max(1.0, abs(Fnew))
            stop_stag = rel_change < ftol_rel

         # max iterations
        stop_maxit = (i >= params["maxit"])


        # ---- termination decision ----
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

    # fallback (should not be reached)
    cnt['iter'] = params['maxit']
    cnt['timetotal'] = time.time() - start_time
    cnt['iflag'] = 1
    return x, cnt

            
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
    if params['useInexactGrad']:
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
        problem.obj_smooth.update(x,'accept')
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

def u_star(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    #base = x*(1-x)*y*(1-y)
    #return (1 + 25*torch.sin(2*math.pi*x)*torch.sin(2*math.pi*y) + 10.0*x*y)
    return torch.sin(math.pi*x)*torch.sin(math.pi*y)

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
    width=32, depth=3, ngrid=32,
    beta=1e-6,
    delta0=1e-1,
    maxit=1,
    device="cpu",
    seed=None,
):
    # model outputs u directly (Dirichlet baked in)
    # Hard boundary conditions
    #model = PoissonNet(width=width, depth=depth).to(device)
    # Soft boundary conditions
    #model = MLP(in_dim=2, width=width, depth=depth, out_dim=1, activation='relu').to(device)
    model = MLPWithFourier(mapping_size=8, scale=0.5, width=64, depth=3, activation="relu")

    # data
    
    xy = make_training_points_grid(ngrid, device=device)
    g = compute_g_from_u_star(xy)  # manufactured forcing on same grid
    # Soft boundary condition
    xb, wb = make_boundary_points(n=ngrid, device=device)
    lam_bc = 100

    # objective + problem wrapper
    var = {
        "useEuclidean": False,
        "beta": beta
    }

    # IMPORTANT: pass kappa_fn, and set mu_I=1e-4 
    obj_smooth = PoissonCompositeObjective(
        model=model,
        xy=xy,
        g=g,
        kappa_fn=kappa_xy,
        weight=None,
        device=device,
        mu_I=0.0,
        xb=xb,
        wb=wb,
        bc_target = torch.zeros((xb.shape[0],1),device=device,dtype=xy.dtype),
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

    # initial parameter vector
    x0 = vector_from_model(model)

    # TR params
    
    params = set_default_parameters("SPG2")
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        params['generator']= generator
    params["delta"] = delta0
    params["maxit"] = maxit
    params["useInexactObj"] = False
    params["useInexactGrad"] = False

    # if you implemented these stopping rules in TR:
    params["pred_floor_rel"] = 1e-12
    params["ared_accept_rel"] = 1e-12
    params["rej_max_stuck_at_floor"] = 10
    #params["batch_size"] = None        # e.g. 256/512/1024; None = full batch
    #params["sample_every"] = 1        # resample every iter
    #x_opt, cnt = trustregion_dropin_stochastic(x0, delta0, problem, params)

    # run TR
    x_opt, cnt = trustregion(x0, delta0, problem, params)

    # load optimum back into model
    load_vector_into_model(x_opt, model)
    return model, x_opt, cnt




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
 
#Hessian check
@torch.no_grad()
def directional_fd_grad_full_locked(obj, theta, v, eps):
    th_p = theta.copy()
    th_m = theta.copy()
    th_p.axpy(+eps, v)
    th_m.axpy(-eps, v)

    # lock mode consistently for both perturbations (important!)
    
    gp, _ = obj.gradient(th_p)
    gm, _ = obj.gradient(th_m)

    out = gp.copy()
    out.axpy(-1.0, gm)
    out.scal(1.0 / (2.0 * eps))
    return out


def hv_check_full(obj, theta, ntests=10, eps_list=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4), gradTol=1e-12):
    torch.set_default_dtype(torch.float64)

    # lock mode at base point
    obj.update(theta, "accept")
    mode = bool(getattr(obj, "_smooth_mode", False))
    print(f"[hv_check_full] locked smooth_mode = {mode}")

    for t in range(ntests):
        v = theta.randn_like()
        v.normalize_()

        # hv at base point, same locked mode
        Hv, _ = obj.hessVec(v, theta, gradTol=gradTol)
        print(f"\nTest {t}: ||Hv|| = {Hv.norm():.6e}")

        for eps in eps_list:
            fdHv = directional_fd_grad_full_locked(obj, theta, v, eps)

            diff = fdHv.copy()
            diff.axpy(-1.0, Hv)

            relerr = diff.norm() / max(1.0, fdHv.norm(), Hv.norm())
            print(f" eps={eps:>8.1e} ||FD-Hv||/scale={relerr:.3e}")

@torch.no_grad()
def gn_quadratic_form_check_interior(obj, theta, ntests=5, eps_list=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    # lock mode at base point
    obj.update(theta, "accept")

    params0 = {k: theta.td[k] for k, _ in obj.model.named_parameters()}
    z = obj._f_of_params_functorch_hybrid(params0).detach()

    for t in range(ntests):
        v = theta.randn_like().normalize_()
        Jv = obj.apply_Jf_functorch(theta, v).detach()

        Bv, _ = obj.hessVec(v, theta)
        vTBv = v.dot(Bv)
        print(f"\nTest {t}: v^T Bv = {vTBv:+.6e}")

        for eps in eps_list:
            hp = obj.h(z + eps * Jv)
            h0 = obj.h(z)
            hm = obj.h(z - eps * Jv)

            fd = (hp - 2.0 * h0 + hm) / (eps ** 2)
            fd = float(fd.detach().cpu().item())
            relerr = abs(fd - vTBv) / max(1.0, abs(fd), abs(vTBv))
            print(f" eps={eps:>8.1e} FD={fd:+.6e} relerr={relerr:.3e}")

@torch.no_grad()
def gn_quadratic_form_check_full(obj, theta, ntests=5, eps_list=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    """
    Check v^T B v for the full Gauss-Newton model:
      interior h(f(theta))
      + boundary penalty
      + optional mu_I * ||step||^2 contribution through hessVec

    This compares hessVec against a second directional derivative of the
    same Gauss-Newton model pieces.
    """
    obj.update(theta, "accept")

    params0 = {k: theta.td[k] for k, _ in obj.model.named_parameters()}
    z0 = obj._f_of_params_functorch_hybrid(params0).detach()

    # boundary state at theta
    ub0 = None
    if (obj.xb is not None) and (obj.lam_bc > 0.0):
        ub0 = obj._ub_of_params_functorch_hybrid(params0).detach()
        target = 0.0 if obj.bc_target is None else obj.bc_target.reshape(-1)

    for t in range(ntests):
        v = theta.randn_like().normalize_()

        Jv = obj.apply_Jf_functorch(theta, v).detach()
        Jubv = None
        if (obj.xb is not None) and (obj.lam_bc > 0.0):
            Jubv = obj.apply_Jub_functorch(theta, v).detach()

        Hv, _ = obj.hessVec(v, theta)
        vTBv = v.dot(Hv)

        print(f"\nTest {t}: v^T Bv = {vTBv:+.6e}")

        for eps in eps_list:
            # interior GN quadratic form
            hp = obj.h(z0 + eps * Jv)
            h0 = obj.h(z0)
            hm = obj.h(z0 - eps * Jv)
            fd_int = (hp - 2.0 * h0 + hm) / (eps ** 2)

            fd_total = fd_int

            # boundary GN quadratic form
            if Jubv is not None:
                rp = (ub0 + eps * Jubv) - target
                r0 = ub0 - target
                rm = (ub0 - eps * Jubv) - target

                if obj.wb is None:
                    bp = 0.5 * obj.lam_bc * torch.mean(rp**2)
                    b0 = 0.5 * obj.lam_bc * torch.mean(r0**2)
                    bm = 0.5 * obj.lam_bc * torch.mean(rm**2)
                else:
                    w = obj.wb.reshape(-1) if obj.wb.ndim > 1 else obj.wb
                    bp = 0.5 * obj.lam_bc * torch.sum(w * (rp**2))
                    b0 = 0.5 * obj.lam_bc * torch.sum(w * (r0**2))
                    bm = 0.5 * obj.lam_bc * torch.sum(w * (rm**2))

                fd_bc = (bp - 2.0 * b0 + bm) / (eps ** 2)
                fd_total = fd_total + fd_bc

            # mu_I contribution
            if obj.mu_I != 0.0:
                fd_total = fd_total + obj.mu_I * (v.norm() ** 2)

            fd_total = float(fd_total.detach().cpu().item())
            relerr = abs(fd_total - vTBv) / max(1.0, abs(fd_total), abs(vTBv))
            print(f" eps={eps:>8.1e} FD={fd_total:+.6e} relerr={relerr:.3e}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    width, depth, ngrid = 32, 4, 64
    mapping_size, scale = 16, 0.5
    beta = 0.0
    model = MLPWithFourier(mapping_size=mapping_size, scale=scale, width=width, depth=depth, activation="relu")
    xy = make_training_points_grid(ngrid, device = device)
    g = compute_g_from_u_star(xy)
    xb, wb = make_boundary_points(n=ngrid, device=device)
    # -------------------------
    # pre-training diagnostics on a fresh model
    # -------------------------
    model0 = MLPWithFourier(
        mapping_size=mapping_size,
        scale=scale,
        width=width,
        depth=depth,
        activation="relu"
    ).to(device)
    obj0 = PoissonCompositeObjective(
        model=model0,
        xy=xy,
        g=g,
        kappa_fn=kappa_xy,
        weight=None,
        device=device,
        mu_I=0.0,
        xb=xb,
        wb=wb,
        bc_target=torch.zeros((xb.shape[0], 1), device=device, dtype=xy.dtype),
        lam_bc=100.0,
    )
