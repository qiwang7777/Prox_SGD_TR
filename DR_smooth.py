import torch
import numpy as np
import copy,time
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
import torch.nn.functional as F
from torch.func import functional_call, grad
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
    params['gtol']    = 1.8e-3
    params['stol']    = 1e-12
    params['ocScale'] = params['t']

    # Trust-region parameters
    params['eta1']     = 0.3
    params['eta2']     = 0.95
    params['gamma1']   = 0.25
    params['gamma2']   = 2.5
    params['delta']    = 1
    params['deltamin'] = 1e-8

    params['deltamax'] = 10

    # Subproblem solve tolerances
    params['atol']    = 1e-5
    params['rtol']    = 1e-3
    params['spexp']   = 2
    params['maxitsp'] = 15

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
    Composite objective: h(f(theta))

    f(theta) = [ vec(grad u_theta(x_i)) ; vec(u_theta(x_i)) ]
    h(z) = mean_i [ 0.5*kappa_i*||grad u_i||^2 - g_i*u_i ]   (if weight=None)

    TR uses:
      - value(theta)
      - gradient(theta)
      - hessVec(v, theta)  (Gauss-Newton in composite form)
    """

    def __init__(self, model, xy, g, kappa_fn, weight=None, device="cpu", mu_I=0.0):
        self.model = model.to(device)
        self.xy = xy.to(device)
        self.g = g.to(device)
        self.kappa_fn = kappa_fn
        self.device = device

        # quadrature weights
        if weight is None:
            self.weight = None
        else:
            w = weight
            if not torch.is_tensor(w):
                w = torch.tensor(w, dtype=self.xy.dtype, device=device)
            self.weight = w.to(device)

        self.mu_I = float(mu_I)
        self._last_theta = None

    def set_mu_I(self, mu_I: float):
        self.mu_I = float(mu_I)

    def update(self, theta, flag: str):
        
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

    # ---------------- f(theta) and h(z) ----------------
    def f(self, theta):
        # NOTE: f(theta) used in value/gradient only (not inside functorch)
        self._set_parameters(theta)

        xy = self.xy.detach().clone().requires_grad_(True)
        u = self.model(xy)  # (N,1)
        grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]  # (N,2)
        return self._pack_f(grad_u, u)

    def h(self, z: torch.Tensor) -> torch.Tensor:
        grad_u, u = self._unpack_f(z)

        kap = self.kappa_fn(self.xy)
        if kap.ndim == 1:
            kap = kap.reshape(-1, 1)

        integrand = 0.5 * kap * (grad_u**2).sum(dim=1, keepdim=True) - self.g * u  # (N,1)

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
            z = self.f(theta)
            val = self.h(z)
        return float(val.detach().cpu().item()), 0.0

    # ---------------- gradient wrt theta ----------------
    def gradient(self, theta, gtol=1e-12):
        self._set_parameters(theta)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            z = self.f(theta)
            val = self.h(z)
            val.backward()

        grad_td = OrderedDict()
        for name, p in self.model.named_parameters():
            grad_td[name] = p.grad.detach().clone()
        return TorchDictVector(grad_td), 0.0
    
    def _u_smooth(self, params, xy, smooth: bool):
        buffers = dict(self.model.named_buffers())
        #if smooth:
        #    return functional_call(self.model, (params, buffers), (xy,), kwargs={"smooth": smooth})
        #else:
        #    return functional_call(self.model, (params, buffers), (xy,))
        return functional_call(self.model, (params, buffers), (xy,), kwargs={"smooth": bool(smooth)})
            
        

    # =========================================================
    # Functorch-safe f_of_params for JVP/VJP
    # =========================================================
    def _f_of_params_functorch(self, params):
        """
        Returns z = [vec(grad u); vec(u)] computed in a functorch-safe way.

        Key: Do NOT call requires_grad_() or torch.autograd.grad inside a transformed fn.
        Use torch.func.grad + vmap instead.
        """
        from torch.func import functional_call, grad, vmap

        xy = self.xy  # (N,2) already a tensor; no requires_grad_ mutation

        def u_single(x_single):
            # x_single: (2,)
            x_single = x_single.unsqueeze(0)  # (1,2)
            u = functional_call(self.model, params, (x_single,))  # (1,1)
            return u.squeeze()  # scalar

        # grad wrt x_single, vectorized over all points
        grad_u = vmap(grad(u_single))(xy)  # (N,2)

        # u for all points (no need to vmap)
        u = functional_call(self.model, params, (xy,))  # (N,1)

        return self._pack_f(grad_u, u)
    def _smooth_act(self,x):
        """
        beta controls how close to ReLu, the larger, the closer
        """
        beta = getattr(self, "smooth_beta", 50.0)
        return F.softplus(beta * x) / beta
    def _f_of_params_functorch_smooth(self, params):
        """
        Returns z = [vec(grad u); vec(u)] computed in a functorch-safe way.

        Key: Do NOT call requires_grad_() or torch.autograd.grad inside a transformed fn.
        Use torch.func.grad + vmap instead.
        """
        xy = self.xy 
        u = self._u_smooth(params, xy, True)
        
        grad_u = grad(lambda x: self._u_smooth(params, x, True).sum())(xy)
        
        
        z = torch.cat([grad_u.reshape(-1), u.reshape(-1)], dim = 0)
        return z
            
            
       

    def apply_Jf_functorch(self, theta, s):
        """
        Exact J_f(theta) s in z-space via jvp
        """
        from torch.func import jvp

        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}
        tang = {k: s.td[k] for k, _ in self.model.named_parameters()}

        z0, Jd = jvp(self._f_of_params_functorch, (params0,), (tang,))
        return Jd

    def apply_JfT_functorch(self, theta, cotangent_z):
        """
        Exact VJP: J_f(theta)^T cotangent_z -> TorchDictVector
        """
        from torch.func import vjp

        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}

        z0, pullback = vjp(self._f_of_params_functorch, params0)
        grads = pullback(cotangent_z)[0]

        hv_td = OrderedDict()
        for name, _ in self.model.named_parameters():
            hv_td[name] = grads[name].detach().clone()
        return TorchDictVector(hv_td)
    def apply_Jf_functorch_smooth(self, theta, s):
        """
        Using softplus as the smoothing of ReLu for approximation of J and will use for Hessian later
        """
        from torch.func import jvp
        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}
        tang = {k: s.td[k] for k, _ in self.model.named_parameters()}
        z0, Jd = jvp(self._f_of_params_functorch_smooth, (params0,), (tang,))
        return Jd
    
    def apply_JfT_functorch_smooth(self, theta, cotangent_z):
        """
        Exact VJP: J_f(theta)^T cotangent_z -> TorchDictVector
        """
        from torch.func import vjp

        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}

        z0, pullback = vjp(self._f_of_params_functorch_smooth, params0)
        grads = pullback(cotangent_z)[0]

        hv_td = OrderedDict()
        for name, _ in self.model.named_parameters():
            hv_td[name] = grads[name].detach().clone()
        return TorchDictVector(hv_td)
    
        

    # ---------------- predicted reduction ----------------
    def predicted_reduction(self, theta, s):
        with torch.enable_grad():
            z = self.f(theta)
            Jd = self.apply_Jf_functorch(theta, s)
            pred = self.h(z) - self.h(z + Jd)
        return float(pred.detach().cpu().item())
    #---------------- Smoothing Hessian ----------------
    def hessVec(self, v, theta, gradTol=1e-12):
        """
        Bv = J_f^T [∇²h(z)] J_f v + mu_I*v
        where J_f, J_f^T are computed using a *smoothed* ReLU surrogate
        ONLY for curvature (HessVec). Everything else unchanged.
        """
        with torch.enable_grad():
            # ---  use smooth J and smooth J^T here ---
            Jv = self.apply_Jf_functorch_smooth(theta, v)  # (M,)

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
            Jv_u    = Jv[N * d:]

            Hz_grad = scale * Jv_grad
            Hz_u    = torch.zeros_like(Jv_u)
            Hz      = torch.cat([Hz_grad.reshape(-1), Hz_u.reshape(-1)], dim=0)

            # ---  use smooth J^T here ---
            hv = self.apply_JfT_functorch_smooth(theta, Hz)

        if self.mu_I != 0.0:
            hv = hv + (self.mu_I * v)
        return hv, 0.0
    
    def gradient_gn_smooth(self, theta):
        params0 = {k: theta.td[k] for k,_ in self.model.named_parameters()}
        z = self._f_of_params_functorch_smooth(params0)
        
        N, d = self.xy.shape
        grad_u, u = self._unpack_f(z)
        kap = self.kappa_fn(self.xy)
        if kap.ndim == 1:
            kap = kap.reshape(-1,1)
        if self.weight is None:
            scale = (1.0/float(N)) * kap
        else:
            if self.weight.numel() == 1:
                scale = self.weight * kap
            else:
                ww = self.weight.reshape(-1, 1) if self.weight.ndim == 1 else self.weight
                scale = ww * kap
        dh_dgrad = (scale * grad_u).reshape(-1)
        if self.weight is None:
            dh_du = (-(1.0/float(N)) * self.g).reshape(-1)
        else:
            if self.weight.numel() == 1:
                dh_du = (-(self.weight) * self.g).reshape(-1)
            else:
                ww = self.weight.reshape(-1, 1) if self.weight.ndim == 1 else self.weight
                dh_du = (-(ww)*self.g).reshape(-1)
        cotangent = torch.cat([dh_dgrad, dh_du], dim=0)
        g = self.apply_JfT_functorch_smooth(theta, cotangent)
        return g
    

    
            
# -------------------------
# 2) ReLU network
# -------------------------   


class MLP(nn.Module):
    """
    Plain MLP: x -> h(f(x))
    Activation can be 'relu' or 'softplus' etc.
    """
    def __init__(self, in_dim=2, width=64, depth=3, out_dim=1, activation="relu"):
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

    def forward(self, x):
        return self.net(x)


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
                h = F.softplus(beta * h) / beta
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
  return s, snorm, pRed, phi, Hs, cnt, params

def trustregion_step_SPG2(x, val,grad, dgrad, phi, problem, params, cnt):
    params.setdefault('maxitsp', 10)
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
        print("DEBUG:","||s||=", snorm,"sHs=", sHs,"||Hs||=", problem.pvector.norm(Hs),"gTs=", problem.pvector.dot(grad, s))
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


def trustregion(x0, Deltai, problem, params):
    start_time = time.time()

    # --- defaults (do not use setdefault here if you want these to always apply) ---
    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 100)
    params.setdefault('gtol', 1.8e-3)
    params.setdefault('stol', 1e-12)
    params.setdefault('ocScale', 1.0)
    params.setdefault('atol', 1e-4)
    params.setdefault('rtol', 1e-2)
    params.setdefault('spexp', 2)
    params.setdefault('eta1', 1e-4)
    params.setdefault('eta2', 0.75)
    params.setdefault('gamma1', 0.25)
    params.setdefault('gamma2', 10.0)
    params.setdefault('delta', Deltai)
    params.setdefault('deltamax', 1e10)
    params.setdefault('reltol', False)

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

    problem.obj_smooth.update(x, "init")

    # --- initial eval ---
    val, _ = problem.obj_smooth.value(x, 1e-12)
    cnt['nobj1'] += 1
    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
    phi = problem.obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    # --- header ---
    print(f"TR method using {params.get('spsolver','SPG2')} Subproblem Solver")
    print("  iter    value    gnorm    del    snorm    nobjs    ngrad    nobjn    nprox     iterSP    flagSP")
    print(f"{0:4d} {0:4d} {val+phi:8.6e} {params['delta']:8.6e}  ---      {cnt['nobj1']:6d}      {cnt['ngrad']:6d}      {cnt['nobj2']:6d}      {cnt['nprox']:6d} --- ---")

    # --- store init ---
    cnt['objhist'].append(val + phi)
    cnt['obj1hist'].append(val)
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

        params['tolsp'] = min(params['atol'], params['rtol'] * (gnorm ** params['spexp']))
        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
            x, val, grad, dgrad, phi, problem, params, cnt
        )

        # evaluate trial
        xnew = x + s
        problem.obj_smooth.update(xnew, 'trial')
        valnew, val_old, cnt = compute_value(xnew, x, val, problem.obj_smooth, pRed, params, cnt)
        phinew_true = problem.obj_nonsmooth.value(xnew)

        # accept/reject
        aRed = (val + phi) - (valnew + phinew_true)
        pRed_val = float(pRed)
        
        print("trial diagnostics:","val=", val, "phi=", phi,"valnew=", valnew, "phi_true=", phinew_true,"pRed=", pRed, "aRed=", aRed)
        pred_floor = max(1e-12, 1e-10 * max(1.0, abs(val + phi)))
        if pRed_val < pred_floor:
            params['delta'] = max(params['deltamin'],params['gamma1'] * params['delta'])
            problem.obj_smooth.update(x,'reject')
            continue
        if pRed > 0 :
            rho = aRed/max(pRed, 1e-16)
        else:
            rho = -np.inf
        print("debug:", "aRed=",float(aRed), "pRed=",(pRed),"rho =", float(rho) if np.isfinite(rho) else rho)

        if rho < params['eta1']:
            # reject
            params['delta'] = max(params['deltamin'], params['gamma1'] * params['delta'])
            problem.obj_smooth.update(x, 'reject')
            
            # keep x, val, phi, grad, dgrad, gnorm unchanged
        else:

            #accept
            x = xnew
            val = valnew
            phi = phinew
            problem.obj_smooth.update(x, 'accept')
            grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
            
            if rho > params['eta2']:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])

        
        if i % params['outFreq'] == 0:
            print(f"{i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      "
                  f"{cnt['nobj1']:6d}     {cnt['ngrad']:6d}       {cnt['nobj2']:6d}     {cnt['nprox']:6d}      "
                  f"{iter_count:4d}        {iflag:1d}")

        # store
        cnt['objhist'].append(val + phi)
        cnt['obj1hist'].append(val)
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
        if (gnorm <= gtol) or (snorm < stol) or (i >= params['maxit']):
            if gnorm <= gtol:
                flag = 0
                reason = "optimality tolerance was met"
            elif i >= params['maxit']:
                flag = 1
                reason = "maximum number of iterations was met"
            else:
                flag = 2
                reason = "step tolerance was met"

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
            grad = problem.obj_smooth.gradient_gn_smooth(x)
            gerr = 0
            cnt['ngrad'] += 1
            dgrad = problem.dvector.dual(grad)
            pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
            cnt['nprox'] += 1
            gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']
            gtol = min(params['maxGradTol'], scale0 * min(gnorm, params['delta']))
    else:
        gtol = 1e-12
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
    base = x*(1-x)*y*(1-y)
    return base*(1 + 25*torch.sin(2*math.pi*x)*torch.sin(2*math.pi*y) + 10*x*y)

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



def train_poisson_with_TR(
    width=32, depth=3, ngrid=32,
    beta=1e-6,
    delta0=1e-1,
    maxit=50,
    device="cpu",
):
    # model outputs u directly (Dirichlet baked in)
    model = PoissonNet(width=width, depth=depth).to(device)

    # data
    xy = make_training_points_grid(ngrid, device=device)
    g = compute_g_from_u_star(xy)  # manufactured forcing on same grid

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
        mu_I=1e-4,
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
    params["delta"] = delta0
    params["maxit"] = maxit
    params["useInexactObj"] = False
    params["useInexactGrad"] = False

    # if you implemented these stopping rules in TR:
    params["pred_floor_rel"] = 1e-12
    params["ared_accept_rel"] = 1e-12
    params["rej_max_stuck_at_floor"] = 10

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
def directional_fd_value(obj, theta, v, eps):
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
    for t in range(ntests):
        v = theta.randn_like()
        v.normalize_()
        gTv = directional_grad_dot(obj, theta, v)
        print(f"\nTest {t}: g^T v = {gTv:+.6e}")
        for eps in eps_list:
            fd = directional_fd_value(obj, theta, v, eps)
            relerr = abs(fd - gTv) / max(1.0, abs(fd), abs(gTv))
            print(f" eps={eps: >8.1e} FD={fd:+.6e} relerr={relerr:.3e}")
 
#Hessian check
@torch.no_grad()
def directional_fd_grad(obj_smooth, theta, v, eps):
    th_p = theta.clone()
    th_m = theta.clone()
    th_p.axpy(eps, v)
    th_m.axpy(-eps, v)
    gp,_ = obj_smooth.gradient(th_p)
    gm,_ = obj_smooth.gradient(th_m)
    out = gp.clone()
    out.axpy(-1.0, gm)
    out.scal(1.0 / (2.0 *eps))
    return out

def directional_fd_grad_gn(obj, theta, v, eps):
    th_p = theta.copy()
    th_m = theta.copy()
    th_p.axpy(eps, v)
    th_m.axpy(-eps, v)
    gp = obj.gradient_gn_smooth(th_p)
    gm = obj.gradient_gn_smooth(th_m)
    out = gp.copy()
    out.axpy(-1.0, gm)
    out.scal(1.0/(2.0*eps))
    return out
    

def hv_check(obj_smooth, theta, ntests = 10, eps_list = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4), gradTol = 1e-12):
    torch.set_default_dtype(torch.float64)
    for t in range(ntests):
        v = theta.randn_like()
        v.normalize_()
        Hv, _ = obj_smooth.hessVec(v, theta, gradTol = gradTol)
        print(f"\nTest {t}: ||Hv|| = {Hv.norm():.6e}")
        for eps in eps_list:
            fdHv = directional_fd_grad_gn(obj_smooth, theta, v, eps)
            num = (fdHv.clone().axpy(-1.0, Hv) or fdHv)
            diff = fdHv.clone()
            diff.axpy(-1.0, Hv)
            relerr = diff.norm() / max(1.0, fdHv.norm(), Hv.norm())
            print(f" eps={eps:>8.1e} ||FD-Hv||/scale={relerr:.3e}")
            
            
@torch.no_grad()
def gn_quadratic_form_check(obj, theta, ntests=5, eps_list=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    params0 = {k: theta.td[k] for k, _ in obj.model.named_parameters()}
    z = obj._f_of_params_functorch_smooth(params0).detach()
    for t in range(ntests):
        v = theta.randn_like().normalize_()
        Jv = obj.apply_Jf_functorch_smooth(theta, v).detach()
        Bv, _ = obj.hessVec(v, theta)
        vTBv = v.dot(Bv)
        print(f"\nTest {t}: v^T Bv ={vTBv:+.6e}")
        for eps in eps_list:
            hp = obj.h(z + eps * Jv)
            h0 = obj.h(z)
            hm = obj.h(z - eps * Jv)
            fd = (hp - 2.0*h0 + hm)/ (eps**2)
            fd = float(fd.detach().cpu().item())
            relerr = abs(fd - vTBv) / (max(1.0, abs(fd), abs(vTBv)))
            print(f" eps={eps:>8.1e} FD={fd:+.6e} relerr = {relerr:.3e}")
    


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    width, depth, ngrid = 32, 2, 32
    beta = 0.0
    model = PoissonNet(width = width, depth = depth).to(device)
    xy = make_training_points_grid(ngrid, device = device)
    g = compute_g_from_u_star(xy)
    var = {"useEuclidean": False, "beta": beta}
    obj_smooth = PoissonCompositeObjective(model = model, xy= xy, g = g, kappa_fn = kappa_xy, weight = None, device = device, mu_I = 1e-4)
    x0 = vector_from_model(model)
    #Derivative check and Hessian check
    print("\n ==== GRAD CHECK at x0 ====")
    grad_check(obj_smooth, x0, ntests=5)
    print("\n ==== HV CHECK at x0 ====")
    gn_quadratic_form_check(obj_smooth, x0, ntests = 3)
    

    model, x_opt, cnt = train_poisson_with_TR(
        width=width,
        depth=depth,
        ngrid=ngrid,
        beta=beta,
        delta0=1e-1,
        maxit= 700,
        device=device,
    )
    

    plot_solution_and_error(model, n=121, device=device)
    plot_tr_history(cnt)
     
