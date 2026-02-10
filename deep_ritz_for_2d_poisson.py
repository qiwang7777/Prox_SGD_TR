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

def vector_from_model(model: torch.nn.Module) -> TorchDictVector:
    td = OrderedDict()
    for name, p in model.named_parameters():
        td[name] = p.detach().clone()
    return TorchDictVector(td)

@torch.no_grad()
def load_vector_into_model(x: TorchDictVector, model: torch.nn.Module):
    name_to_param = dict(model.named_parameters())
    for k, v in x.td.items():
        name_to_param[k].copy_(v)

def ensure_same_structure(x: TorchDictVector, model: torch.nn.Module):
    model_keys = [n for n, _ in model.named_parameters()]
    x_keys = list(x.td.keys())
    assert model_keys == x_keys, "TorchDictVector keys must match model.named_parameters() order."


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
    params['gtol']    = 1e-8
    params['stol']    = 1e-12
    params['ocScale'] = params['t']

    # Trust-region parameters
    params['eta1']     = 0.05
    params['eta2']     = 0.9
    params['gamma1']   = 0.25
    params['gamma2']   = 2.5
    params['delta']    = 1
    params['deltamin'] = 1e-8

    params['deltamax'] = 1e5

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
    return 1.1 + 0.2*torch.sin(2*math.pi*x)*torch.cos(2*math.pi*y)

def u_star(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    base = x*(1-x)*y*(1-y)
    return base*(1 + 0.25*torch.sin(2*math.pi*x)*torch.sin(2*math.pi*y) + 0.1*x*y)

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
    
    def hessVec(self, v:TorchDictVector, x: TorchDictVector, gradTol=1e-12):
        #B_k v = mu*v, where we picked B_k=mu*I
        hv_td = OrderedDict()
        for k,vk in v.td.items():
            hv_td[k]=self.mu * vk
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
        g0s    = problem.pvector.dot(g0_primal,s)
        phinew = problem.obj_nonsmooth.value(x1)
        alpha0 = -(g0s + phinew - phiold) / sHs
        if sHs <= params['safeguard']: 
          alpha = alphamax
        else:
          alpha = np.minimum(alphamax,alpha0)
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


# -------------------------
# 3) Trust-region method
# -------------------------

    




def trustregion(x0, Deltai, problem, params):
    start_time = time.time()

    # --- defaults (do not use setdefault here if you want these to always apply) ---
    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 100)
    params.setdefault('gtol', 1e-6)
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
        problem.obj_smooth.set_mu(1.0 )#/ max(params['delta'], 1e-6))

        params['tolsp'] = min(params['atol'], params['rtol'] * (gnorm ** params['spexp']))
        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
            x, val, grad, dgrad, phi, problem, params, cnt
        )

        # evaluate trial
        xnew = x + s
        problem.obj_smooth.update(xnew, 'trial')
        valnew, val_old, cnt = compute_value(xnew, x, val, problem.obj_smooth, pRed, params, cnt)

        # accept/reject
        aRed = (val + phi) - (valnew + phinew)
        if pRed > 0 :
            rho = aRed/pRed
        else:
            rho = -np.inf
        print("debug:", "aRed=",float(aRed), "pRed=",(pRed),"rho =", float(rho) if np.isfinite(rho) else rho)

        if aRed < params['eta1'] * pRed:
            # reject
            params['delta'] = max(params['deltamin'], params['gamma1'] * params['delta'])
            problem.obj_smooth.update(x, 'reject')
            # keep x, val, phi, grad, dgrad, gnorm unchanged
        else:
            # accept
            x = xnew
            val = valnew
            phi = phinew
            problem.obj_smooth.update(x, 'accept')
            grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)

            if aRed > params['eta2'] * pRed:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])

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

        # stopping (this is the ONLY place we terminate)
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
    params["delta"] = delta0
    params["maxit"] = maxit
    params["useInexactObj"] = False
    params["useInexactGrad"] = False
    params["beta"] = beta  # convenience

    # call your TR driver
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
        width=128,
        depth=4,
        ngrid=32,
        beta=1e-6,
        delta0=1e-1,
        maxit=500,
        device=device,
    )

    err = rel_l2_error_on_grid(model, n=64, device=device)
    plot_solution_and_error(model, n=121, device=device)
    plot_tr_history(cnt)

    print("Relative L2 error on grid:", err)
