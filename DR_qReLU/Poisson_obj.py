import torch
import numpy as np
from torch.func import functional_call, grad, jvp, vjp, vmap
from collections import OrderedDict
import math
from semismooth_TR.TorchVector import TorchDictVector

# --- boundary factor for homogeneous Dirichlet ---
def b_factor(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return x*(1-x)*y*(1-y)

def u_star(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    base = x*(1-x)*y*(1-y)
    return base*(1 + 25*torch.sin(2*math.pi*x)*torch.sin(2*math.pi*y) + 10*x*y)

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

    def __init__(self, model, xy, g, kappa_fn, weight=None, device="cpu", mu_I=0.0,
                 xb=None, wb=None, bc_target=None, lam_bc=0.0, x_true=None,u_true_fn=None):
        self.model = model.to(device)
        self.xy = xy.to(device)
        self.g = g.to(device)
        self.kappa_fn = kappa_fn
        self.device = device
        self.xb = xb.to(device) if xb is not None else None
        self.wb = wb.to(device) if wb is not None else None
        self.bc_target = bc_target.to(device) if bc_target is not None else None
        self.lam_bc = float(lam_bc)
        self.hess_mode = "full"
        self.x_true=x_true
        self.u_true_fn=u_true_fn
            

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
        
    def set_hess_mode(self, mode: str):
        if mode not in ("gn", "full"):
            raise ValueError("mode must be 'gn' or 'full'")
        self.hess_mode = mode

    def update(self, theta, flag: str):
        
        self._last_theta = None
        if hasattr(self, "xy_full"):
            self.xy=self.xy_full
        if hasattr(self,"g_full"):
            self.g = self.g_full
        if hasattr(self,"weight_full"):
            self.weight = self.weight_full

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
    
    def _f_of_params_functorch(self, params):
        """
        Returns z = [vec(grad u); vec(u)] computed in a functorch-safe way.

        """

       
        
        
        xy = self.xy  # (N,2) already a tensor; no requires_grad_ mutation
        buffers = dict(self.model.named_buffers())

        def u_single(x_single):
            # x_single: (2,)
            x_single = x_single.unsqueeze(0)  # (1,2)
            u = functional_call(self.model, (params,buffers), (x_single,))  # (1,1)
            return u.squeeze()  # scalar

        # grad wrt x_single, vectorized over all points
        grad_u = vmap(grad(u_single))(xy)  # (N,2)

        # u for all points (no need to vmap)
        u = functional_call(self.model, (params,buffers), (xy,))  # (N,1)

        return self._pack_f(grad_u, u)
    
   
    


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
        
    def _value_of_params_functorch(self, params):
        z = self._f_of_params_functorch(params)
        return self.h(z)

    # ---------------- objective value ----------------
    def value(self, theta, ftol=1e-12):
        params0={k:theta.td[k] for k, _ in self.model.named_parameters()}
        val = self._value_of_params_functorch(params0)
        return float(val.detach().cpu().item()), 0.0

    # ---------------- gradient wrt theta ----------------
    def gradient(self, theta, gtol=1e-12):
        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}
        grad_fn = grad(self._value_of_params_functorch)
        grads = grad_fn(params0)

        grad_td = OrderedDict()
        for name, _ in self.model.named_parameters():
            grad_td[name] = grads[name].detach().clone()
    
        return TorchDictVector(grad_td), 0.0

    # =========================================================
    # Functorch-safe f_of_params for JVP/VJP
    # =========================================================
    
    
    
    def hessVec_full_ad(self, v, theta, gradTol=1e-12):
        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}
        tang = {k: v.td[k] for k, _ in self.model.named_parameters()}
        
        grad_fn = grad(self._value_of_params_functorch)
        _, hvp = jvp(grad_fn, (params0,), (tang,))
        
        hv_td = OrderedDict()
        for name, _ in self.model.named_parameters():
            hv_td[name] = hvp[name].detach().clone()
            
        hv = TorchDictVector(hv_td)
        if self.mu_I != 0.0:
            hv = hv + self.mu_I * v
        return hv, 0.0

            
            
       

    def apply_Jf_functorch(self, theta, s):
        """
        Exact J_f(theta) s in z-space via jvp
        """
        

        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}
        tang = {k: s.td[k] for k, _ in self.model.named_parameters()}

        z0, Jd = jvp(self._f_of_params_functorch, (params0,), (tang,))
        return Jd

    def apply_JfT_functorch(self, theta, cotangent_z):
        """
        Exact VJP: J_f(theta)^T cotangent_z -> TorchDictVector
        """
        

        params0 = {k: theta.td[k] for k, _ in self.model.named_parameters()}

        z0, pullback = vjp(self._f_of_params_functorch, params0)
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
    def hessVec_gn(self, v, theta, gradTol=1e-12):
        """
        Bv = J_f^T [∇²h(z)] J_f v + mu_I*v
        """
        with torch.enable_grad():
            # ---  interior GN curvature ---
            Jv = self.apply_Jf_functorch(theta, v)  # (M,)

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
            
            hv = self.apply_JfT_functorch(theta,Hz)

        if self.mu_I != 0.0:
            hv = hv + (self.mu_I * v)
        return hv, 0.0
    
    def hessVec(self,v, theta, gradTol=1e-12):
        if self.hess_mode == "gn":
            return self.hessVec_gn(v, theta, gradTol)
        elif self.hess_mode == "full":
            return self.hessVec_full_ad(v, theta, gradTol)
        else:
            raise ValueError(f"Unknown hess_mode:{self.hess_mode}")
    
    def attach_smoother(self,smoother):
        self.smoother = smoother
        
    def value_smooth(self, theta, mu, ftol=1e-12):
        if not hasattr(self, "smoother"):
            raise RuntimeError("No smoother has been attached.")
        return self.smoother.value(theta,mu,ftol)
    
    def gradient_smooth(self,theta,mu,gtol=1e-12):
        if not hasattr(self,"smoother"):
            raise RuntimeError("No smoother has been attached.")
        return self.smoother.gradient(theta, mu, gtol)
    
    def hessVec_smooth(self, v, theta, mu, gradTol=1e-12):
        if not hasattr(self, "smoother"):
            raise RuntimeError("No smoother has been attached.")
        return self.smoother.hessvec(theta, v, mu, gradTol) 
    
    

    
    # relative L2 error diagnostic
    def relative_L2_error(self, theta):
        """
        Compute relative L2 error 
        ||u_theta - u_star||/||u_star||
        using uniform grid quadrature

        """
        if self.u_true_fn is None:
            return np.inf
        self._set_parameters(theta)
        with torch.no_grad():
            xy = self.xy
            u_pred = self.model(xy)
            u_true = u_star(xy)
            N = xy.shape[0]
            n_side = int(N ** 0.5)
            h = 1.0 / (n_side-1)
            weight = h * h
            err_sq = weight * torch.sum((u_pred-u_true)**2)
            true_sq = weight * torch.sum(u_true ** 2)
            rel_L2 = torch.sqrt(err_sq / true_sq)
        return rel_L2.item()
    
