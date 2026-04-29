import torch

class EuclideanPrimal:
    def __init__(self, var=None):
        self.var = var

    @torch.no_grad()
    def dot(self, x, y):
        return float(torch.sum(x.data * y.data).item())

    @torch.no_grad()
    def norm(self, x):
        return float(torch.sqrt(torch.sum(x.data * x.data)).item())


class EuclideanDual:
    def __init__(self, var=None):
        self.var = var

    @torch.no_grad()
    def apply(self, x, y):
        return float(torch.sum(x.data * y.data).item())

    @torch.no_grad()
    def dual(self, x):
        return x


class L2TVPrimal(EuclideanPrimal):
    pass


class L2TVDual(EuclideanDual):
    pass
