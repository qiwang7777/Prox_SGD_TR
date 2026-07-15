import torch
from .ControlVector import ControlVector

class L1Penalty:
    def __init__(self, var):
        self.beta = float(var["beta"]); self.weight = float(var["weight"])
    @torch.no_grad()
    def value(self, x):
        return float(self.beta * self.weight * torch.sum(torch.abs(x.data)).item())
    @torch.no_grad()
    def prox(self, x, t):
        th = float(t) * self.beta * self.weight
        z = torch.sign(x.data) * torch.clamp(torch.abs(x.data) - th, min=0.0)
        return ControlVector(z)
    def get_parameter(self): return self.beta
