import torch
from collections import OrderedDict
from semismooth_TR.TorchVector import TorchDictVector

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
    
