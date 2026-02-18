import torch
import numpy as np
import copy,time
import torch.nn as nn
import matplotlib.pyplot as plt
import math

# -------------------------
# 0) ReLu network and parameters set up
# -------------------------

from collections import OrderedDict

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


#Derivative check    
def td_inner(x: TorchDictVector, y: TorchDictVector) -> float:
    s = 0.0
    for k in x.td.keys():
        s = s + torch.sum(x.td[k] * y.td[k]).item()
    return float(s)

def td_rand_like(x: TorchDictVector, seed = 0) -> TorchDictVector:
    torch.manual_seed(seed)
    out = TorchDictVector()
    for k, v in x.td.items():
        out.td[k] = torch.randn_like(v)
    return out

def td_norm(x: TorchDictVector) -> float:
    return np.sqrt(td_inner(x,x))

#@torch.no_grad()
def check_gradient(obj_smooth, x: TorchDictVector, ntests = 3, seed = 0):
    """
    obj_smooth: PoissonEnergyObjective
    x: TorchDictVector parameters
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    f0, _ = obj_smooth.value(x)
    g, _ = obj_smooth.gradient(x)
    print("\n====Gradient check (directional derivative) ====")
    print("f(x) =", f0)
    print("||g|| = ", td_norm(g))
    base_steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    for t in range(ntests):
        v = td_rand_like(x, seed = seed + 1234 + t)
        v = (1.0 / max(td_norm(v), 1e-30)) * v
        gv = td_inner(g, v)
        print(f"\nTest direction {t}: <g,v>={gv: .16e}")
        print("eps fd(central)  error(|fd-gv|)  ration(prev/now)")
        
        prev_err = None
        for eps in base_steps:
            xp = x + eps * v
            xm = x - eps * v
            fp, _ = obj_smooth.value(xp)
            fm, _ = obj_smooth.value(xm)
            
            fd = (fp - fm) / (2.0 * eps)
            err = abs(fd - gv)
            ratio = (prev_err / err) if (prev_err is not None and err > 0 ) else np.nan
            print(f"{eps:1.0e}  {fd:+.16e}   {err:.3e}   {ratio:8.3f}")

#update for mu
def estimate_fd_curvature_scale(obj_smooth, x, nprobe=5, eps=1e-5, seed=0):
    g0, _ = obj_smooth.gradient(x)
    vals = []
    for t in range(nprobe):
        v = td_rand_like(x, seed=seed + 1000 + t)
        nv = td_norm(v)
        if nv < 1e-30:
            continue
        v = (1.0 / nv) * v
        gp, _ = obj_smooth.gradient(x + eps * v)
        dg = gp - g0
        c_fd = td_inner(dg, v) / eps
        if np.isfinite(c_fd):
            vals.append(abs(c_fd))
    if len(vals) == 0:
        return 1e-3
    return float(np.median(vals))

class LBFGSHessianCompact:
    """
    Stable L-BFGS approximation of the Hessian B_k (not inverse).

    - Stores (s_i, y_i) pairs
    - Uses compact representation to apply B_k v robustly:
        B = gamma I + W M^{-1} W^T
      where W = [Y, gamma S], and M is a 2m x 2m matrix (Nocedal-Wright).

    - Uses Powell damping to keep curvature (s^T y) sufficiently positive.
    """

    def __init__(self, m=10, curv_eps=1e-14, damp=True, damp_c=0.2):
        self.m = int(m)
        self.curv_eps = float(curv_eps)
        self.damp = bool(damp)
        self.damp_c = float(damp_c)

        self.S = []   # list[TorchDictVector]
        self.Y = []   # list[TorchDictVector]
        self.gamma = 1.0

    def reset(self):
        self.S.clear()
        self.Y.clear()
        self.gamma = 1.0

    def _dot(self, a, b):
        # expects TorchDictVector
        s = 0.0
        for k, v in a.td.items():
            s = s + torch.sum(v * b.td[k]).item()
        return float(s)

    def _scale(self, a, x):
        out = TorchDictVector()
        for k, v in x.td.items():
            out.td[k] = a * v
        return out

    def _axpy(self, a, x, y):
        out = TorchDictVector()
        for k in x.td.keys():
            out.td[k] = a * x.td[k] + y.td[k]
        return out

    def _matvec_Wt(self, v):
        """
        Compute W^T v where W = [Y, gamma S]
        Returns numpy array length 2m:
          [y1^T v, ..., ym^T v, (gamma s1)^T v, ..., (gamma sm)^T v]
        """
        m = len(self.S)
        out = np.zeros(2*m, dtype=np.float64)
        for i in range(m):
            out[i] = self._dot(self.Y[i], v)
        for i in range(m):
            out[m+i] = self.gamma * self._dot(self.S[i], v)
        return out

    def _matvec_W(self, z):
        """
        Compute W z where W = [Y, gamma S]
        z is numpy array length 2m.
        Returns TorchDictVector.
        """
        m = len(self.S)
        out = self.S[0].zero_like() #if m > 0 else v.zero_like()  # safe fallback

        # Y part
        for i in range(m):
            out = self._axpy(float(z[i]), self.Y[i], out)
        # gamma S part
        for i in range(m):
            out = self._axpy(float(z[m+i]) * self.gamma, self.S[i], out)
        return out

    def _build_M(self):
        """
        Build the 2m x 2m compact matrix:
          M = [[-D,  -L^T],
               [-L,  gamma*S^T S]]
        where:
          D = diag(s_i^T y_i)
          L = strictly lower part of S^T Y (i>j)
          S^T S is Gram matrix of S
        """
        m = len(self.S)
        STY = np.zeros((m, m), dtype=np.float64)
        STS = np.zeros((m, m), dtype=np.float64)

        for i in range(m):
            for j in range(m):
                STY[i, j] = self._dot(self.S[i], self.Y[j])
                STS[i, j] = self._dot(self.S[i], self.S[j])

        D = np.diag(np.diag(STY))
        L = np.tril(STY, k=-1)

        top = np.hstack([-D, -L.T])
        bot = np.hstack([-L, self.gamma * STS])
        M = np.vstack([top, bot])

        return M

    def apply(self, v):
        """
        Return B_k v using compact representation:
          Bv = gamma v + W * (M^{-1} * (W^T v))
        """
        m = len(self.S)
        if m == 0:
            return self._scale(self.gamma, v)

        # base term
        r = self._scale(self.gamma, v)

        # compact correction
        wtv = self._matvec_Wt(v)         # shape (2m,)
        M = self._build_M()              # shape (2m,2m)

        # solve M z = W^T v
        try:
            z = np.linalg.solve(M, wtv)
        except np.linalg.LinAlgError:
            # fallback: just gamma*I if M is ill-conditioned
            return r

        Wz = self._matvec_W(z)
        r = r + Wz
        return r

    def update(self, s, y):
        """
        Add (s,y) with curvature checks.
        Optional Powell damping:
          ensure s^T y >= c * s^T B s
        """
        # raw curvature
        sty = self._dot(s, y)
        if (not np.isfinite(sty)) or (sty <= self.curv_eps):
            return

        # update gamma (Hessian scaling)
        ss = self._dot(s, s)
        if np.isfinite(ss) and ss > self.curv_eps:
            self.gamma = max(self.curv_eps, sty / ss)

        if self.damp and len(self.S) > 0:
            # Powell damping needs s^T B s
            Bs = self.apply(s)
            sBs = self._dot(s, Bs)

            if (np.isfinite(sBs) and sBs > self.curv_eps):
                # if curvature too small, damp y
                if sty < self.damp_c * sBs:
                    theta = (1.0 - self.damp_c) * sBs / (sBs - sty)
                    # y <- theta y + (1-theta) B s
                    y = self._axpy((1.0 - theta), Bs, self._scale(theta, y))
                    sty2 = self._dot(s, y)
                    if (not np.isfinite(sty2)) or (sty2 <= self.curv_eps):
                        return
                    sty = sty2

        # store copies (detach/clone)
        self.S.append(s.copy())
        self.Y.append(y.copy())

        if len(self.S) > self.m:
            self.S.pop(0)
            self.Y.pop(0)
class LBFGSHessian:
    """
    Limited-memory BFGS approximation for the Hessian B_k.
    
    Stores (s_i,y_i) pairs from accepted steps:
        s_i = x_{i+1} - x_{i}
        y_i = g_{i+1} - g_{i}
        
    Provides:
        -apply(v) : returns B_k v
        -update(s, y): adds a new correction pair (with curvature checks)
        
    Uses initial B0 = gamma * I (gamma estimated from latest pair).
    """
    def __init__(self, m=10, curvature_eps=1e-12):
        self.m = int(m)
        self.curv_eps = float(curvature_eps)
        self.S = [] #list[TorchDictVector]
        self.Y = [] #list[TorchDictVector]
        self.sy = [] #list[float]
        self.gamma = 1.0
        
    def reset(self):
        self.S.clear()
        self.Y.clear()
        self.sy.clear()
        self.gamma = 1.0
        
    def update(self, s: TorchDictVector, y: TorchDictVector):
        sy = td_dot(s,y)
        if not np.isfinite(sy) or sy <= self.curv_eps:
            # skip bad curvatire pairs
            return
        yy = td_dot(y, y)
        if np.isfinite(yy) and yy> self.curv_eps:
            #common scaling: gamma ~ (y^T y)/(s^T y) (H0^{-1} scaling);
            #for B0 = gamma I, a stable choice is gamma ~ (s^T y)/(s^T s)
            ss = td_dot(s, s)
            if np.isfinite(ss) and ss>self.curv_eps:
                self.gamma = sy / ss #B0 scaling
        
        # push, keep limited memory
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.sy.append(float(sy))
        
        if len(self.S) > self.m:
            self.S.pop(0); self.Y.pop(0); self.sy.pop(0)
            
    def _apply_prefix(self, v: TorchDictVector, upto: int) -> TorchDictVector:
        """
        Apply BFGS updates 0..up to 1 to vector v, starting from B0 = gamma*I.
        This is used internally to compute (B_{i},s_i)
        """
        r = td_scale(self.gamma, v) #r = B0 v
        for j in range(upto):
            s = self.S[j]
            y = self.Y[j]
            sy = self.sy[j]
            if sy <= self.curv_eps:
                continue
            Bs = self._apply_prefix(s,j)
            sBs = td_dot(s, Bs)
            if (not np.isfinite(sBs)) or sBs <= self.curv_eps:
                continue
            sTr = td_dot(s, r) #s^T (B_j v)
            yTv = td_dot(y, v) #y^T v (needs original v)
            
            #B_{j+1} v = B_j v-(B_j s s^T B_j v)/(s^T B_j s)+(y y^T v)/(y^T s)
            r = td_axpy(-sTr / sBs, Bs, r)
            r = td_axpy(yTv / sy, y, r)
        return r
    
    def apply(self, v: TorchDictVector) -> TorchDictVector:
        if len(self.S) == 0:
            return td_scale(self.gamma, v)
        
        return self._apply_prefix(v, len(self.S))
            
#Hessian sanity check (for LBFGS)
#"SPD" CHECK if v.T B v > 0
def check_lbfgs_spd(obj_smooth, x: TorchDictVector, ntests=50, seed=0):
    """
    Sanity-check the model Hessian used by TR:
        Hv = obj_smooth.hessVec(v, x) = (LBFGS + mu I) v
    Prints v^T Hv statistics and flags negatives / crazy magnitudes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    vals = []
    neg = 0
    bad = 0

    for t in range(ntests):
        v = td_rand_like(x, seed=seed + 1000 + t)
        nv = td_norm(v)
        if nv < 1e-30:
            continue
        v = (1.0 / nv) * v

        Hv, _ = obj_smooth.hessVec(v, x, 1e-12)
        q = td_inner(v, Hv)

        if (not np.isfinite(q)):
            bad += 1
            continue
        if q < 0:
            neg += 1
        vals.append(q)

    if len(vals) == 0:
        print("No valid samples (all NaN/Inf?)")
        return

    vals_np = np.array(vals, dtype=float)
    print("\n==== LBFGS+muI model SPD sanity ====")
    print(f"samples={len(vals)}, neg={neg}, bad={bad}")
    print(f"min(v^T Hv) = {vals_np.min():+.3e}")
    print(f"median       = {np.median(vals_np):+.3e}")
    print(f"max          = {vals_np.max():+.3e}") 

def check_directional_curvature(obj_smooth, x: TorchDictVector, ntests=10, eps=1e-5, seed=0):
    """
    Compare directional curvature from gradient differences vs. model curvature:
        c_fd = <g(x+eps v)-g(x), v>/eps
        c_model = <Bv, v>  where Bv = hessVec(v)
    For ReLU this can jump if activations flip; keep eps small and expect some outliers.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    g0, _ = obj_smooth.gradient(x)

    print("\n==== Directional curvature: FD-grad vs model (LBFGS+muI) ====")
    print("test   c_fd            c_model         ratio(c_fd/c_model)")
    for t in range(ntests):
        v = td_rand_like(x, seed=seed + 3000 + t)
        nv = td_norm(v)
        if nv < 1e-30:
            continue
        v = (1.0 / nv) * v

        # FD curvature
        gp, _ = obj_smooth.gradient(x + eps * v)
        dg = gp - g0
        c_fd = td_inner(dg, v) / eps

        # model curvature
        Bv, _ = obj_smooth.hessVec(v, x, 1e-12)
        c_model = td_inner(Bv, v)

        ratio = c_fd / c_model if (np.isfinite(c_fd) and np.isfinite(c_model) and abs(c_model) > 1e-30) else np.nan
        print(f"{t:2d}   {c_fd:+.6e}   {c_model:+.6e}    {ratio:+.3e}")       
def check_directional_curvature_consistent(obj, x, pvector, ntests=10, eps=1e-5, seed=0):
    """
    Compare:
      c_fd    = <g(x+eps v) - g(x-eps v), v> / (2 eps)
      c_model = <H_model v, v>

    IMPORTANT: normalize v in the SAME norm used by your solver (pvector.norm).
    """
    rng = np.random.default_rng(seed)

    g0, _ = obj.gradient(x)

    print("\n==== Directional curvature: FD-grad vs model (consistent scaling) ====")
    print("test   c_fd            c_model         ratio(c_fd/c_model)")

    for t in range(ntests):
        # random direction in parameter space
        v = td_rand_like(x, seed=int(rng.integers(0, 10**9)))  # you implement this

        # normalize in solver norm
        nv = pvector.norm(v)
        if not np.isfinite(nv) or nv < 1e-14:
            continue
        v = (1.0 / nv) * v

        # finite-diff of gradient
        x_p = x + (eps * v)
        x_m = x - (eps * v)

        gp, _ = obj.gradient(x_p)
        gm, _ = obj.gradient(x_m)

        dg = gp - gm
        c_fd = pvector.dot(dg, v) / (2.0 * eps)

        # model curvature
        Hv, _ = obj.hessVec(v, x, gradTol=1e-12)
        c_model = pvector.dot(Hv, v)

        ratio = c_fd / c_model if (np.isfinite(c_model) and abs(c_model) > 1e-14) else np.nan

        print(f"{t:2d}   {c_fd:+.6e}   {c_model:+.6e}   {ratio:+.3e}")

class ReLUMLP(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

        # optional: small init helps with stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in = m.weight.size(1)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        return self.net(x)

def b_factor(xy):
    x, y = xy[:,0:1], xy[:,1:2]
    return x*(1-x)*y*(1-y)

def u_theta(model, xy):
    return b_factor(xy) * (1 + model(xy))

def relu_signatures(model: nn.Module, xy: torch.Tensor):
    """
    Returns a list of boolean tensors indicating pre-activation > 0
    for each ReLU layer (based on linear outputs before ReLU).
    Assumes Sequential [Linear, ReLU, Linear, ReLU, ..., Linear].
    """
    sigs = []
    h = xy
    layers = list(model.net)
    for i in range(len(layers)):
        h = layers[i](h)
        if isinstance(layers[i], nn.ReLU):
            # this is after relu, so we need pre-activation. Instead, record sign before applying relu.
            # We'll handle by peeking previous output: easiest is restructure, but here's a workaround:
            pass
    return sigs

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
    params['gtol']    = 1e-4
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

class Problem:
    def __init__(self, var, R):
        self.var  = var
        if hasattr(var, '__getitem__'):
            Euclid_check = var['useEuclidean']
        else:
            Euclid_check = True
        if Euclid_check:
            self.pvector   = Euclidean(var)
            self.dvector   = Euclidean(var)
        else:
            self.pvector   = L2TVPrimal(var)
            self.dvector   = L2TVDual(var)

class L2TVPrimal:
    def __init__(self, var):
        self.var = var
    @torch.no_grad()
    def dot(self, x, y):
        ans = 0
        for k, v in x.td.items():
            ans += torch.sum(torch.mul(v, y.td[k]))
        return ans.item()
    @torch.no_grad()
    def apply(self, x, y):
        return self.dot(x, y)
    @torch.no_grad()
    def norm(self, x):
        return np.sqrt(self.dot(x, x))
    @torch.no_grad()
    def dual(self, x):
        return x
#L2TVDual for NN
class L2TVDual:
    def __init__(self, var):
        self.var = var
    @torch.no_grad()
    def dot(self, x, y):
        ans = 0
        for k, v in x.td.items():
            ans += torch.sum(torch.mul(v, y.td[k]))
        return ans.item()
    @torch.no_grad()
    def apply(self, x, y):
        return self.dot(x,y)
    @torch.no_grad()
    def norm(self, x):
        return np.sqrt(self.dot(x, x))
    @torch.no_grad()
    def dual(self, x):
        return x
    
class Euclidean:
    def __init__(self, var):
        self.var = var

    def dot(self, x, y):
        return x.T @ y

    def apply(self, x, y):
        return x.T @ y

    def norm(self, x):
        return np.sqrt(self.dot(x, x))

    def dual(self, x):
        return x
    
class L1TorchNorm:
    def __init__(self, var):
        self.var = var
        
    @torch.no_grad()
    def value(self, x):
        val = 0
        for _,v in x.td.items():
            val += torch.sum(torch.abs(v)).item()
        return float(self.var['beta']*val)

    #def value(self, x):
    #    val = 0
    #    for k, v in x.td.items():
    #        val += torch.sum(torch.abs(v))
    #    return self.var['beta'] * val
    
    def prox(self, x, t):
        temp = x.clone()
        beta = self.var['beta']
        for k, v in x.td.items():
            shrink = torch.clamp(torch.abs(v) - t*beta, min = 0.0)
            temp.td[k] = shrink * torch.sign(v)
        return temp

    #def prox(self, x, t):
    #    temp = x.clone()
    #    for k, v in x.td.items():
    #        temp.td[k] = torch.max(torch.tensor([0.0]), torch.abs(v) - t*self.var['beta'])*torch.sign(v)
    #    # return np.maximum(0, np.abs(x) - t * self.var['Rlump'] * self.var['beta']) * np.sign(x)
    #    return temp

    def dir_deriv(self, s, x):
        sx = np.sign(x)
        return self.var['beta'] * (np.dot(sx.T, s) + np.dot((1 - np.abs(sx)).T, np.abs(s)))

    def project_sub_diff(self, g, x):
        sx = np.sign(x)
        return self.var['beta'] * sx + (1 - np.abs(sx)) * np.clip(g, -self.var['beta'], self.var['beta'])

    def gen_jac_prox(self, x, t):
        d = np.ones_like(x)
        px = self.prox(x, t)
        ind = px == 0
        d[ind] = 0
        return np.diag(d), ind

    def apply_prox_jacobian(self, v, x, t):
        if self.var['useEuclidean']:
            ind = np.abs(x) <= t * self.var['Rlump'] * self.var['beta']
        else:
            ind = np.abs(x) <= t * self.var['beta']
        Dv = v.copy()
        Dv[ind] = 0
        return Dv
    def subdiff(self,x):
        subdiff_result = {}
        for k,v in x.td.items():
            subdiff_result[k] = torch.where(
                v>0, torch.tensor(1.0),
                torch.where(v<0,torch.tensor(-1.0),torch.tensor([-1.0,1.0]))
            )


        return subdiff_result

    def get_parameter(self):
        return self.var['beta']
    


# -------------------------
# 1) Set up model
# -------------------------

# g


def kappa_xy(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    #return 1.0 #+ 0.2*torch.sin(2*math.pi*x)*torch.cos(2*math.pi*y)
    return torch.ones_like(x)

def u_star(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    base = x*(1-x)*y*(1-y)
    return base*(1 + 25*torch.sin(2*math.pi*x)*torch.sin(2*math.pi*y) + 15*x*y)

def compute_g_from_u_star(xy: torch.Tensor) -> torch.Tensor:
    """
    g(x) = -div( kappa(x) * grad u*(x) )
    computed by AD.
    """
    xy_req = xy.detach().clone().requires_grad_(True)

    u = u_star(xy_req)                       # (N,1)
    grad_u = torch.autograd.grad(
        u.sum(), xy_req, create_graph=True
    )[0]                                     # (N,2)

    kap = kappa_xy(xy_req)                   # (N,1)
    flux = kap * grad_u                      # broadcasting -> (N,2)

    div_flux = 0.0
    for j in range(2):
        div_flux = div_flux + torch.autograd.grad(
            flux[:, j].sum(), xy_req, create_graph=True
        )[0][:, j:j+1]                       # (N,1)

    g = -div_flux.detach()
    return g

# Obj_smooth (actually semismooth)

class PoissonEnergyObjective:
    """
    Smooth part:
        J_smooth(θ) = ∫_Ω [ 1/2 κ(x) |∇u_θ(x)|^2 - g(x) u_θ(x) ] dx
    approximated by Monte-Carlo / grid average.

    Required interface for your TR framework:
      - update(x, flag)
      - value(x, ftol) -> (float, err)
      - gradient(x, gtol) -> (TorchDictVector, err)
      - hessVec(v, x, gradTol) -> (TorchDictVector, err)
    """
    def __init__(self, model, xy, g, device="cpu", mu=1e-3):
        self.model = model.to(device)
        self.xy = xy.to(device)
        self.g = g.to(device)
        self.device = device
        self.mu = mu

        # internal cache (optional)
        self._last_x = None
        self._last_val = None
        
        # Hessian LBFGS
        #self.lbfgs = LBFGSHessian(m = 10, curvature_eps = 1e-14)
        self.lbfgs = LBFGSHessianCompact(m = 10, curv_eps = 1e-14, damp = True, damp_c = 0.2)

    def update(self, x: TorchDictVector, flag: str):
        # keep it simple; you can add caching based on flag if you want
        self._last_x = None
        self._last_val = None
        
    def set_mu(self,mu):
        self.mu = float(mu)

    def _energy(self) -> torch.Tensor:
        """
        Returns scalar tensor (smooth energy) for current model params.
        """
        xy = self.xy.detach().clone().requires_grad_(True)

        u = u_theta(self.model, xy)                 # (N,1)
        grad_u = torch.autograd.grad(
            u.sum(), xy, create_graph=True
        )[0]                                        # (N,2)

        kap = kappa_xy(xy)                          # (N,1)
        integrand = 0.5 * kap * (grad_u**2).sum(dim=1, keepdim=True) - self.g * u

        # Ω=(0,1)^2 has area 1, so integral ≈ mean(integrand)
        return integrand.mean()

    def value(self, x: TorchDictVector, ftol=1e-12):
        ensure_same_structure(x, self.model)
        load_vector_into_model(x, self.model)

        with torch.enable_grad():
            val = self._energy()

        # framework expects (val, err)
        return float(val.detach().cpu().item()), 0.0

    def gradient(self, x: TorchDictVector, gtol=1e-12):
        ensure_same_structure(x, self.model)
        load_vector_into_model(x, self.model)

        # enable grads on params
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.model.zero_grad(set_to_none=True)

        val = self._energy()
        val.backward()

        grad_td = OrderedDict()
        for name, p in self.model.named_parameters():
            grad_td[name] = p.grad.detach().clone()

        return TorchDictVector(grad_td), 0.0

    #def hessVec(self, v: TorchDictVector, x: TorchDictVector, gradTol=1e-12):
    #    """
    #    Hessian-vector product of the smooth part, at x, applied to v.
    #    Implemented via double backward:
    #      Hv = ∇( <∇J(x), v> )
    #    """
    #    ensure_same_structure(x, self.model)
    #    load_vector_into_model(x, self.model)

    #    # make params require grad for 2nd derivatives
    #    params = []
    #    names = []
    #    for name, p in self.model.named_parameters():
    #        p.requires_grad_(True)
    #        params.append(p)
    #        names.append(name)

    #    self.model.zero_grad(set_to_none=True)

    #    with torch.enable_grad():
    #        val = self._energy()
    #        g_list = torch.autograd.grad(val, params, create_graph=True)

    #        # inner product <g, v>
    #        inner = 0.0
    #        for gi, name in zip(g_list, names):
    #            inner = inner + (gi * v.td[name]).sum()

    #        hv_list = torch.autograd.grad(inner, params, retain_graph=False, create_graph=False)

    #    hv_td = OrderedDict()
    #    for name, hvi in zip(names, hv_list):
    #        hv_td[name] = hvi.detach().clone()

    #    return TorchDictVector(hv_td), 0.0
    
    #Hess = mu*I
    #def hessVec(self, v:TorchDictVector, x: TorchDictVector, gradTol=1e-12):
    #    #B_k v = mu*v, where we picked B_k=mu*I
    #    hv_td = OrderedDict()
    #    for k,vk in v.td.items():
    #        hv_td[k]=self.mu * vk
    #    return TorchDictVector(hv_td), 0.0
    
    def hessVec(self, v: TorchDictVector, x: TorchDictVector, gradTol=1e-12):
        #B_k v ~ (L-BFGS Hessian) v+ mu * v (damping keeps SPD-ish)
        Bv = self.lbfgs.apply(v)
        nv = td_norm(v)
        nb = td_norm(Bv)
        if np.isfinite(nv) and nv > 1e-30 and np.isfinite(nb):
            cap = 100.0 * (self.mu + 1.0)
            if nb > cap *nv:
                scale = (cap*nv)/(nb+1e-30)
                for k in Bv.td.keys():
                    Bv.td[k] = scale * Bv.td[k]
        
        hv_td = OrderedDict()
        for k in v.td.keys():
            hv_td[k] = Bv.td[k] + self.mu * v.td[k]
        return TorchDictVector(hv_td), 0.0
        

    # Optional hooks your TR driver checks for
    def begin_counter(self, i, cnt):  # if you want
        return cnt
    def end_counter(self, i, cnt):
        return cnt

   


# -------------------------
# 2) Subsolver
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

  #if sHs > params['safeguard']: #*problem.pvector.dot(s,s):
  ss = problem.pvector.dot(s, s)
  curv_tol = params['safeguard'] * max(1.0, ss)
  if (np.isfinite(sHs)) and (sHs > curv_tol):
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

def trustregion_step_SPG2(x, val, grad, dgrad, phi, problem, params, cnt):
    """
    
    Returns:
        s      : step from TR center x (i.e., x_trial - x)
        snorm  : ||s||
        pRed   : predicted reduction (model)
        phinew : nonsmooth value at x_trial
        iflag  : 0 converged, 1 maxitsp hit, 2 hit TR boundary
        iter   : iterations used
        cnt, params
    """
    # -------------------------
    # defaults
    # -------------------------
    params.setdefault('maxitsp', 10)
    params.setdefault('lam_min', 1e-12)
    params.setdefault('lam_max', 1e12)
    params.setdefault('t', 1.0)
    params.setdefault('gradTol', np.sqrt(np.finfo(float).eps))

    params.setdefault('safeguard', np.sqrt(np.finfo(float).eps))
    params.setdefault('atol', 1e-4)
    params.setdefault('rtol', 1e-2)
    params.setdefault('spexp', 2)

    # -------------------------
    # init (SPG2 iterate x0 starts at TR center x)
    # -------------------------
    x0 = x.copy()
    g0 = grad.copy()          # model gradient at x0 (updated by adding H*s steps)
    t0 = params.get('t_gcp', params['t'])

    # model bookkeeping
    valold = val
    phiold = phi
    phinew = phiold

    # -------------------------
    # Start from generalized Cauchy point (1 prox step)
    # -------------------------
    sc, snormc, _, _, _, cnt, params = trustregion_gcp2(x, val, grad, dgrad, phi, problem, params, cnt)
    s = sc.copy()             # first direction (already within TR ball)
    x1 = x0 + s               # trial for phinew etc.

    # stopping tol for inner loop (same spirit as your original)
    gnorm_dir = max(problem.pvector.norm(s), 0.0)
    gtol = min(params['atol'], params['rtol'] * (gnorm_dir / max(t0, 1e-16)) ** params['spexp'])

    iflag = 1
    iters = 0

    # -------------------------
    # main SPG2 loop
    # -------------------------
    for k in range(1, params['maxitsp'] + 1):
        iters = k

        # incremental direction for this step is always
        #   d = x1 - x0
        d = x1 - x0
        dd = problem.pvector.dot(d, d)
        if dd <= 0.0:
            # no movement possible
            iflag = 0
            break

        # total step so far from TR center
        s_total = x0 - x
        ss_total = problem.pvector.dot(s_total, s_total)

        # -------------------------
        # TR boundary: compute alphamax so ||s_total + alpha*d|| <= delta
        # -------------------------
        alphamax = 1.0
        if ss_total >= (1.0 - params['safeguard']) * (params['delta'] ** 2):
            # already (nearly) on boundary; only allow tangential / shrink
            alphamax = 0.0
        else:
            # if full step would exit ball, clip
            s_next = s_total + d
            if problem.pvector.dot(s_next, s_next) >= (1.0 - params['safeguard']) * (params['delta'] ** 2):
                ds = problem.pvector.dot(d, s_total)
                rad = ds * ds + dd * (params['delta'] ** 2 - ss_total)
                alphamax = min(1.0, (-ds + np.sqrt(max(rad, 0.0))) / max(dd, 1e-16))

        # -------------------------
        # curvature along direction d (not along total step)
        # -------------------------
        Hd, _ = problem.obj_smooth.hessVec(d, x, params['gradTol'])
        # cnt['nhess'] += 1  # if you track it
        dHd = problem.dvector.apply(Hd, d)
        g0d = problem.pvector.dot(g0, d)

        # nonsmooth at full x1 (used in alpha0)
        phitmp = problem.obj_nonsmooth.value(x1)
        cnt['nobj2'] += 1

        # curvature tolerance scaled by ||d||^2
        curv_tol = params['safeguard'] * max(1.0, dd)

        # -------------------------
        # choose alpha
        # -------------------------
        if (not np.isfinite(dHd)) or (dHd <= curv_tol):
            alpha = alphamax
        else:
            alpha0 = -(g0d + phitmp - phiold) / dHd
            alpha = min(alphamax, alpha0)

        # -------------------------
        # apply step alpha*d to x0 (update model gradient and model value)
        # -------------------------
        if alpha <= 0.0:
            # cannot move without leaving TR ball
            iflag = 2
            break

        x0 = x0 + alpha * d
        s_total = x0 - x
        snorm = problem.pvector.norm(s_total)

        # model gradient update: g <- g + alpha*H*d
        g0 = g0 + alpha * problem.dvector.dual(Hd)

        # model value update
        valold = valold + alpha * g0d + 0.5 * (alpha ** 2) * dHd

        # nonsmooth at new x0
        phinew = problem.obj_nonsmooth.value(x0)
        cnt['nobj2'] += 1
        phiold = phinew

        # -------------------------
        # stop if hit TR boundary
        # -------------------------
        if snorm >= (1.0 - params['safeguard']) * params['delta']:
            iflag = 2
            break

        # -------------------------
        # spectral step length update (BB-like)
        # -------------------------
        if (not np.isfinite(dHd)) or (dHd <= 0.0):
            lambdaTmp = params['t'] / max(problem.pvector.norm(g0), 1e-16)
        else:
            lambdaTmp = dd / dHd

        t0 = max(params['lam_min'], min(params['lam_max'], lambdaTmp))

        # -------------------------
        # next prox step (defines next direction d = x1 - x0)
        # -------------------------
        x1 = problem.obj_nonsmooth.prox(x0 - t0 * g0, t0)
        cnt['nprox'] += 1

        # convergence check
        d_next = x1 - x0
        gnorm_dir = problem.pvector.norm(d_next)
        if (gnorm_dir / max(t0, 1e-16) <= gtol):
            iflag = 0
            break

    # final outputs
    s = x0 - x
    snorm = problem.pvector.norm(s)
    pRed = (val + phi) - (valold + phinew)

    return s, snorm, float(pRed), float(phinew), int(iflag), int(iters), cnt, params


# -------------------------
# 3) Trust-region method
# -------------------------

    




def trustregion(x0, Deltai, problem, params):
    

    start_time = time.time()

    # -----------------------------
    # Defaults
    # -----------------------------
    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 200)
    params.setdefault('gtol', 1e-4)
    params.setdefault('stol', 1e-12)
    params.setdefault('ocScale', 1.0)
    params.setdefault('reltol', False)

    # TR parameters
    params.setdefault('eta1', 0.3)
    params.setdefault('eta2', 0.95)
    params.setdefault('gamma1', 0.25)
    params.setdefault('gamma2', 2.5)

    params.setdefault('delta', Deltai)
    params.setdefault('deltamin', 1e-8)
    params.setdefault('deltamax', 10.0)

    # Subproblem tolerances
    params.setdefault('atol', 1e-5)
    params.setdefault('rtol', 1e-3)
    params.setdefault('spexp', 2)

    # mu control
    params.setdefault('mu', 1e-3)
    params.setdefault('mu_min', 1e-6)
    params.setdefault('mu_max', 1e3)

    # FD curvature refresh
    params.setdefault('mu_fd_freq', 20)
    params.setdefault('mu_fd_eps', 1e-5)
    params.setdefault('mu_fd_nprobe', 5)

    # Reject handling / recovery
    params.setdefault('rej_reset_lbfgs_after', 5)
    params.setdefault('rej_max_stuck_at_floor', 10)

    # IMPORTANT: larger floor so rho isn't computed from meaningless pRed.
    # In your logs pRed ~ 1e-8..1e-6, so 1e-12 is too small.
    params.setdefault('pred_floor_rel', 1e-8)     # << changed from 1e-12
    params.setdefault('ared_accept_rel', 1e-12)

    # Recovery-mode knobs
    params.setdefault('recovery_len', 20)         # iterations to run with mu*I only
    params.setdefault('recovery_delta', 1e-6)     # relax delta to escape kink micro-regime

    # -----------------------------
    # Counters/history
    # -----------------------------
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

    # -----------------------------
    # Helpers
    # -----------------------------
    def _lbfgs_reset():
        lb = getattr(problem.obj_smooth, "lbfgs", None)
        if lb is not None and hasattr(lb, "reset"):
            lb.reset()

    def _set_mu(mu):
        mu = float(np.clip(mu, params['mu_min'], params['mu_max']))
        params['mu'] = mu
        problem.obj_smooth.set_mu(mu)

    def _estimate_mu_fd(i, x):
        cscale = estimate_fd_curvature_scale(
            problem.obj_smooth, x,
            nprobe=params['mu_fd_nprobe'],
            eps=params['mu_fd_eps'],
            seed=i
        )
        # modest mu ~ cscale (NO extra 10x here)
        return float(np.clip(cscale, params['mu_min'], params['mu_max']))

    # recovery-mode state
    recovery_left = 0

    def _enter_recovery(i, x, why=""):
        nonlocal recovery_left
        if why:
            print(f"[recovery] enter @ iter {i}: {why}")
        _lbfgs_reset()

        # Switch to mu*I-only model (requires tiny change in objective; see below)
        if hasattr(problem.obj_smooth, "use_lbfgs"):
            problem.obj_smooth.use_lbfgs = False

        mu_fd = _estimate_mu_fd(10_000 + i, x)
        _set_mu(mu_fd)

        # relax delta a bit to escape “kink micro-regime”
        params['delta'] = min(params['deltamax'], max(params['delta'], params['recovery_delta']))
        recovery_left = int(params['recovery_len'])

    def _maybe_leave_recovery(i):
        nonlocal recovery_left
        if recovery_left > 0:
            recovery_left -= 1
            if recovery_left == 0:
                if hasattr(problem.obj_smooth, "use_lbfgs"):
                    problem.obj_smooth.use_lbfgs = True
                # after recovery, we keep current mu (already FD-scaled)

    def maybe_update_mu_from_fd(i, x):
        # Only update mu from FD when not in recovery
        if recovery_left > 0:
            return
        if i == 1 or (params['mu_fd_freq'] > 0 and (i % params['mu_fd_freq'] == 0)):
            _set_mu(_estimate_mu_fd(i, x))

    def maybe_hessian_sanity(i, x):
        if not params.get("do_hess_sanity", False):
            return
        freq = params.get("hess_sanity_freq", 50)
        if freq <= 0 or (i % freq != 0):
            return
        obj = problem.obj_smooth
        lb = getattr(obj, 'lbfgs', None)
        m = len(lb.S) if (lb is not None and hasattr(lb, "S")) else 0
        print(f"\n==== Hessian sanity @ iter {i} (LBFGS mem={m}, mu={obj.mu}) ====")
        check_lbfgs_spd(obj, x, ntests=30, seed=i)
        check_directional_curvature(obj, x, ntests=10, eps=1e-5, seed=i)
        print("==== end sanity ====\n")

    # -----------------------------
    # init x
    # -----------------------------
    if params['initProx']:
        x = problem.obj_nonsmooth.prox(x0, 1.0)
        cnt['nprox'] += 1
    else:
        x = x0.copy()

    problem.obj_smooth.update(x, "init")

    # initial eval
    val, _ = problem.obj_smooth.value(x, 1e-12)
    cnt['nobj1'] += 1
    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
    phi = problem.obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    # header
    print(f"TR method using {params.get('spsolver','SPG2')} Subproblem Solver")
    print("  iter    value    gnorm    del    snorm    nobjs    ngrad    nobjn    nprox     iterSP    flagSP")
    print(f"{0:4d} {0:4d} {val+phi:8.6e} {params['delta']:8.6e}  ---      "
          f"{cnt['nobj1']:6d}      {cnt['ngrad']:6d}      {cnt['nobj2']:6d}      {cnt['nprox']:6d} --- ---")

    # store init
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

    # stopping tolerances
    gtol = params['gtol']
    stol = params['stol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm
        stol = params['stol'] * gnorm

    # reject / stuck tracking
    rej_count = 0
    floor_stuck = 0

    # -----------------------------
    # main loop
    # -----------------------------
    for i in range(1, params['maxit'] + 1):

        # recovery bookkeeping
        _maybe_leave_recovery(i)

        # update mu occasionally (FD-scaled)
        maybe_update_mu_from_fd(i, x)
        _set_mu(params['mu'])

        # solve TR subproblem
        params['tolsp'] = min(params['atol'], params['rtol'] * (gnorm ** params['spexp']))
        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
            x, val, grad, dgrad, phi, problem, params, cnt
        )

        # trial eval
        xnew = x + s
        dx = problem.pvector.norm(xnew - x)
        print(f"[sanity] ||xnew-x|| = {dx:.3e}, delta={params['delta']:.3e}, snorm={snorm:.3e}")
        problem.obj_smooth.update(xnew, 'trial')
        #valnew, _, cnt = compute_value(xnew, x, val, problem.obj_smooth, pRed, params, cnt)
        valnew, _ = problem.obj_smooth.value(xnew, 1e-12)
        cnt['nobj1'] += 1

        # actual + predicted reduction
        aRed = (val + phi) - (valnew + phinew)
        pRed = float(pRed)

        # scale-aware floors
        fscale = abs(val + phi) + 1.0
        pred_floor = params['pred_floor_rel'] * fscale
        ared_floor = params['ared_accept_rel'] * fscale

        # compute rho only when pred is meaningful
        if (np.isfinite(pRed) and (abs(pRed) > pred_floor)):
            rho = float(aRed / pRed)
        else:
            rho = np.nan

        print("debug:", "aRed=", float(aRed), "pRed=", pRed,
              "rho =", (float(rho) if np.isfinite(rho) else rho),
              "mu=", params['mu'], "delta", params['delta'])

        # -----------------------------
        # Accept/reject decision
        # -----------------------------
        # If pRed tiny: accept based on actual decrease only (rho unreliable).
        if (not np.isfinite(pRed)) or (abs(pRed) <= pred_floor):
            accept = (np.isfinite(aRed) and (aRed > ared_floor))
        else:
            accept = (np.isfinite(rho) and (rho >= params['eta1']))

        if not accept:
            rej_count += 1
            

            # shrink delta
            params['delta'] = max(params['deltamin'], params['gamma1'] * params['delta'])
            problem.obj_smooth.update(x, 'reject')

            # reset LBFGS after many rejects
            if rej_count >= params['rej_reset_lbfgs_after']:
                _lbfgs_reset()
                rej_count = 0

            # floor logic
            if params['delta'] <= params['deltamin'] * 1.0001:
                floor_stuck += 1

                # if objective change is basically within noise: stop
                if (not np.isfinite(aRed)) or (abs(aRed) <= 100.0 * ared_floor):
                    print("Terminating: delta at minimum and objective change is within numerical noise.")
                    cnt['iter'] = i
                    cnt['timetotal'] = time.time() - start_time
                    cnt['iflag'] = 3
                    return x, cnt

                # real increase at floor => enter recovery (mu*I only + relax delta)
                _enter_recovery(i, x, why="stuck at deltamin with real increase (ReLU kink / noisy regime)")

                if floor_stuck >= params['rej_max_stuck_at_floor']:
                    print("Terminating: repeatedly stuck near deltamin with true objective increase (nonsmooth/noisy regime).")
                    cnt['iter'] = i
                    cnt['timetotal'] = time.time() - start_time
                    cnt['iflag'] = 4
                    return x, cnt

            continue  # rejected

        # -----------------------------
        # Accept step
        # -----------------------------
        rej_count = 0
        floor_stuck = 0

        x_old = x
        grad_old = grad

        x = xnew
        val = valnew
        phi = phinew
        problem.obj_smooth.update(x, 'accept')

        grad_new, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)

        # LBFGS update only on accepted steps AND only when enabled
        lb = getattr(problem.obj_smooth, 'lbfgs', None)
        use_lbfgs = getattr(problem.obj_smooth, "use_lbfgs", True)
        if use_lbfgs and (lb is not None) and hasattr(lb, 'update'):
            s_vec = x - x_old
            y_vec = grad_new - grad_old
            if problem.pvector.norm(s_vec) > 1e-14:
                lb.update(s_vec, y_vec)

        grad = grad_new

        # TR radius update ONLY if rho meaningful
        if np.isfinite(rho) and (rho >= params['eta2']):
            params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])

        maybe_hessian_sanity(i, x)

        # print iter
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

    # fallback
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
            grad, gerr = problem.obj_smooth.gradient(x, gtol)
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
# 4) Training model
# -------------------------
def make_training_points_grid(n=32, device="cpu"):
    xs = torch.linspace(0.0, 1.0, n, device=device)
    ys = torch.linspace(0.0, 1.0, n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return xy

def train_poisson_with_TR(
    width=128, depth=4, ngrid=32,
    beta=1e-6,
    delta0=1e-1,
    maxit=50,
    device="cpu",
):
    # model
    model = ReLUMLP(width=width, depth=depth).to(device)

    # data
    xy = make_training_points_grid(ngrid, device=device)
    g = compute_g_from_u_star(xy)  # manufactured forcing

    # objective + problem wrapper
    var = {
        "useEuclidean": False,  # we use the dict-vector inner products below
        "beta": beta
    }
    obj_smooth = PoissonEnergyObjective(model=model, xy=xy, g=g, device=device)
    obj_nonsmooth = L1TorchNorm(var)

    # Build a "problem-like" object with fields your solvers call:
    class _ProblemWrap:
        pass
    problem = _ProblemWrap()
    problem.obj_smooth = obj_smooth
    problem.obj_nonsmooth = obj_nonsmooth

    # vectors (dot/norm)
    problem.pvector = L2TVPrimal(var)
    problem.dvector = L2TVDual(var)

    # initial parameter vector
    x0 = vector_from_model(model)

    # TR params
    params = set_default_parameters("SPG2")
    params["mu"] = 1e-3
    params["mu_min"] = 1e-6
    params["mu_max"] = 1e2
    params["mu_fd_freq"] = 20
    params["mu_fd_eps"] = 1e-5
    params["mu_fd_nprobe"] = 5

    params["pred_floor_rel"] = 1e-12
    params["ared_accept_rel"] = 1e-12

    params["rej_reset_lbfgs_after"] = 5
    params["rej_max_stuck_at_floor"] = 10
    
    params["delta"] = delta0
    params["maxit"] = maxit
    params["useInexactObj"] = False
    params["useInexactGrad"] = False
    params["beta"] = beta  # convenience
    params["do_hess_sanity"] = True
    params["hess_sanity_freq"] = 50   # or 20

    # call your TR driver
    
    #check_gradient(obj_smooth, x0, ntests=2, seed=0)
    check_lbfgs_spd(obj_smooth, x0, ntests=50, seed=0)
    check_directional_curvature(obj_smooth, x0, ntests=10, eps=1e-5, seed=0)
    x_opt, cnt = trustregion(x0, delta0, problem, params)

    # load optimum back into model
    load_vector_into_model(x_opt, model)

    return model, x_opt, cnt

# Training

@torch.no_grad()
def rel_l2_error_on_grid(model, n=64, device="cpu"):
    xy = make_training_points_grid(n, device=device)
    u_pred = u_theta(model, xy)
    u_ref = u_star(xy)
    num = torch.norm(u_pred - u_ref)
    den = torch.norm(u_ref)
    return float((num/den).cpu().item())

# -------------------------
# 5) Plotting the results
# -------------------------


@torch.no_grad()
def plot_solution_and_error(model, n=101, device="cpu"):
    # grid
    xs = torch.linspace(0.0, 1.0, n, device=device)
    ys = torch.linspace(0.0, 1.0, n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    # fields
    u_pred = u_theta(model, xy).reshape(n, n).detach().cpu().numpy()
    u_ref  = u_star(xy).reshape(n, n).detach().cpu().numpy()
    err    = u_pred - u_ref

    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()

    # plots
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
    # Safely pull histories if they exist
    obj = np.array(cnt.get("objhist", []), dtype=float)
    gnm = np.array(cnt.get("gnormhist", []), dtype=float)
    delt = np.array(cnt.get("deltahist", []), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    if len(obj) > 0:
        axes[0].plot(obj)
        axes[0].set_title("Objective (val + phi)")
        axes[0].set_xlabel("iter")
    else:
        axes[0].set_title("Objective history missing")

    if len(gnm) > 0:
        axes[1].plot(gnm)
        axes[1].set_title("gnorm (prox-grad norm)")
        axes[1].set_xlabel("iter")
        axes[1].set_yscale("log")
    else:
        axes[1].set_title("gnorm history missing")

    if len(delt) > 0:
        axes[2].plot(delt)
        axes[2].set_title("Trust-region radius delta")
        axes[2].set_xlabel("iter")
        axes[2].set_yscale("log")
    else:
        axes[2].set_title("delta history missing")

    plt.show()



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    

    model, x_opt, cnt = train_poisson_with_TR(
        width=32,
        depth=3,
        ngrid=32,
        beta=0,
        delta0=1e-1,
        maxit=5000,
        device=device,
    )

    err = rel_l2_error_on_grid(model, n=64, device=device)
    plot_solution_and_error(model, n=121, device=device)
    plot_tr_history(cnt)

    print("Relative L2 error on grid:", err)
