import torch
import numpy as np
from .ControlVector import ControlVector
class IndicatorBox:
    """
    phi(u) = I_{C_ad}(u), C_ad = {u_a <= u <= u_b}
    """
    def __init__(self, var):
        self.var = var

    @torch.no_grad()
    def value(self, x):
        u_a = float(self.var["u_a"])
        u_b = float(self.var["u_b"])
        ok = torch.all(x.data >= u_a) and torch.all(x.data <= u_b)
        return 0.0 if ok else np.inf

    @torch.no_grad()
    def prox(self, x, t):
        u_a = float(self.var["u_a"])
        u_b = float(self.var["u_b"])
        return ControlVector(torch.clamp(x.data, min=u_a, max=u_b))

    def get_parameter(self):
        return float(self.var["u_a"]), float(self.var["u_b"])