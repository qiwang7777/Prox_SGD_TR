import torch
import numpy as np
class SaturationLeastSquares:
    def __init__(self, b, A_apply, AT_apply, x_true=None, key="img", kink_tol=1e-8, use_reactivation=True,reactivation_only_when_all_saturated=True):
        self.b = b
        self.A_apply = A_apply
        self.AT_apply = AT_apply
        self.x_true = x_true
        self.key = key
        self.kink_tol = kink_tol
        self.use_reactivation = use_reactivation
        self.reactivation_only_when_all_saturated = reactivation_only_when_all_saturated

        # placeholders used in TR code
        self.xy = None
        self.g = None
        self.weight = None
        self.V = None
        self.dV = None

        self.xy_full = None
        self.g_full = None
        self.weight_full = None
        self.V_full = None
        self.dV_full = None
        
        self._Ax = None
        self._SAx = None
        self._res = None
        self._mask_cache = None
        self._react_v = None
        self._react_mask = None
        self._using_reactivation = False
        
        

    @staticmethod
    def S(z):
        return torch.clamp(z, 0.0, 1.0)

    def _mask(self, z):
        return ((z > self.kink_tol) & (z < 1.0 - self.kink_tol)).to(z.dtype)
    
    def _reactivation_vector(self, z):
        v = torch.zeros_like(z)
        v = torch.where(z < 0.0, z, v)
        v = torch.where(z > 1.0, z - 1.0, v)
        return v
    
    def _reactivation_mask(self,z):
        return ((z < 0.0) | (z > 1.0)).to(z.dtype)
    
    def _should_use_reactivation(self,free_mask):
        if not self.use_reactivation:
            return False
        if self.reactivation_only_when_all_saturated:
            return bool(torch.count_nonzero(free_mask).item() == 0)
        return bool(torch.count_nonzero(free_mask).item() == 0)
            
        
        
            

    def update(self, x, mode="accept"):
        u = x.td[self.key]
        self._Ax = self.A_apply(u)
        self._SAx = self.S(self._Ax)
        self._res = self._SAx - self.b
        self._mask_cache = self._mask(self._Ax)
        
        self._react_v = self._reactivation_vector(self._Ax)
        self._react_mask = self._reactivation_mask(self._Ax)
        self._using_reactivation = self._should_use_reactivation(self._mask_cache)
        

    @torch.no_grad()
    def value(self, x, tol=1e-12):
        u = x.td[self.key]
        Ax = self.A_apply(u)
        SAx = self.S(Ax)
        res = SAx - self.b
        val = 0.5 * torch.sum(res * res)
        return float(val.item()), 0.0

    @torch.no_grad()
    def value_model(self, x, tol=1e-12):
        u = x.td[self.key]
        Ax = self.A_apply(u)
        mask = self._mask(Ax)
        
        if self._should_use_reactivation(mask):
            v = self._reactivation_vector(Ax)
            val = 0.5 * torch.sum(v * v)
            return float(val.item()), 0.0
        SAx = self.S(Ax)
        res = SAx - self.b
        val = 0.5 * torch.sum(res * res)
        #return self.value(x, tol)
        return float(val.item()), 0.0

    @torch.no_grad()
    def gradient(self, x, tol=1e-12):
        u = x.td[self.key]
        Ax = self.A_apply(u)
        mask = self._mask(Ax)
        g = x.zero_like()
        if self._should_use_reactivation(mask):
            v = self._reactivation_vector(Ax)
            g_img = self.AT_apply(v)
            g.td[self.key] = g_img
            return g, 0.0
        SAx = self.S(Ax) 
        res = SAx - self.b
        
        g_img = self.AT_apply(mask*res)
        g.td[self.key] = g_img
        return g, 0.0



    @torch.no_grad()
    def hessVec(self, s, x, tol=1e-12):
        u = x.td[self.key]
        Ax = self.A_apply(u)
        mask = self._mask(Ax)
        sv = s.td[self.key]
        As = self.A_apply(sv)
        Hs = s.zero_like()
        if self._should_use_reactivation(mask):
            react_mask = self._reactivation_mask(Ax)
            Hs_img = self.AT_apply(react_mask * As)
            Hs.td[self.key] = Hs_img
            return Hs, 0.0
        Hs_img = self.AT_apply(mask*As)
        Hs.td[self.key] = Hs_img
        return Hs, 0.0
    
    


    @torch.no_grad()
    def relative_L2_error(self, x):
        if self.x_true is None:
            return np.nan
        u = x.td[self.key]
        num = torch.norm(u - self.x_true)
        den = torch.norm(self.x_true) + 1e-16
        return float((num / den).item())
    
    @torch.no_grad()
    def diagnostics(self,x):
        u = x.td[self.key]
        Ax = self.A_apply(u)
        free = ((Ax > self.kink_tol) &(Ax < 1.0 - self.kink_tol)).sum().item()
        below = (Ax < 0.0).sum().item()
        above = (Ax > 1.0).sum().item()
        at_lower = torch.abs(Ax) <= self.kink_tol
        at_upper = torch.abs(Ax - 1.0) <= self.kink_tol
        kink = (at_lower | at_upper).sum().item()
        return {
            "free": int(free),
            "below_0": int(below),
            "above_1": int(above),
            "at_kink": int(kink),
            "using_reactivation":bool(self._should_use_reactivation(self._mask(Ax))),
            }
        
