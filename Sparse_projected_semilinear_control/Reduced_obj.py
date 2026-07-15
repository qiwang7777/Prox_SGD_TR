import torch
from .ControlVector import ControlVector
from .pde_solver import solve_state, solve_adjoint, solve_linearized_state

class ProjectedSemilinearControlObjective:
    def __init__(self,A,y_d,alpha=1e-4,a=0.0,b=0.4,weight=1.0,
                 zero_selection=0.5,newton_tol=1e-10,newton_maxit=50):
        self.A=A; self.y_d=y_d; self.alpha=float(alpha); self.a=float(a); self.b=float(b)
        self.weight=float(weight); self.zero_selection=float(zero_selection)
        self.newton_tol=float(newton_tol); self.newton_maxit=int(newton_maxit)
        self.x_true=None; self.hess_mode="gauss_newton"; self.mu_I=0.0
        self._state=None; self._trial_state=None
    def set_mu_I(self,mu_I): self.mu_I=float(mu_I)
    def set_hess_mode(self,mode): self.hess_mode=mode
    def update(self,x,flag):
        if flag=="accept" and self._trial_state is not None:
            self._state=self._trial_state.detach().clone()
        elif flag=="reject": self._trial_state=None
    def _state_for(self,x):
        y,_,ok=solve_state(x.data,self.A,self.a,self.b,y0=self._state,
                           tol=self.newton_tol,maxit=self.newton_maxit,
                           zero_selection=self.zero_selection)
        if not ok: raise RuntimeError("State semismooth Newton solve did not converge.")
        self._trial_state=y.detach().clone()
        return y
    def value(self,x,ftol=1e-12):
        y=self._state_for(x); m=y-self.y_d
        val=0.5*self.weight*torch.sum(m*m)+0.5*self.alpha*self.weight*torch.sum(x.data*x.data)
        return float(val.item()),0.0
    def value_model(self,x,ftol=1e-12): return self.value(x,ftol)
    def gradient(self,x,gtol=1e-12):
        y=self._state_for(x)
        p=solve_adjoint(y,self.y_d,self.A,self.a,self.b,self.zero_selection)
        return ControlVector((self.weight*(p+self.alpha*x.data)).detach().clone()),0.0
    def hessVec(self,v,x,gradTol=1e-12):
        y=self._state_for(x)
        z=solve_linearized_state(v.data,y,self.A,self.a,self.b,self.zero_selection)
        w=solve_linearized_state(z,y,self.A,self.a,self.b,self.zero_selection)
        return ControlVector((self.weight*(w+self.alpha*v.data)).detach().clone()),0.0
    @torch.no_grad()
    def active_fractions(self,x,tol=1e-8):
        y=self._state_for(x)
        low=torch.mean((y<=self.a+tol).to(y.dtype))
        mid=torch.mean(((y>self.a+tol)&(y<self.b-tol)).to(y.dtype))
        high=torch.mean((y>=self.b-tol).to(y.dtype))
        return float(low),float(mid),float(high)
    @torch.no_grad()
    def relative_L2_error(self,x): return float("inf")
