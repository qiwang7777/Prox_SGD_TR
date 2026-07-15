import torch

from .GridVector import GridVector


class L1Penalty:
    """
    phi(u) = beta * integral_Omega |u(x)| dx.

    On the uniform grid this is approximated by

        beta * weight * sum_i |u_i|.

    The proximal map is componentwise soft thresholding.
    """

    def __init__(self, var):
        self.beta = float(var["beta"])
        self.weight = float(var["weight"])

    @torch.no_grad()
    def value(self, x):
        return float(
            self.beta * self.weight * torch.sum(torch.abs(x.data)).item()
        )

    @torch.no_grad()
    def prox(self, x, t):
        threshold = float(t) * self.beta * self.weight
        data = torch.sign(x.data) * torch.clamp(
            torch.abs(x.data) - threshold,
            min=0.0,
        )
        return GridVector(data)

    def get_parameter(self):
        return self.beta
