import torch

class EuclideanPrimal:
    def __init__(self, var=None): self.var = {} if var is None else var
    @torch.no_grad()
    def dot(self, x, y): return float(torch.sum(x.data * y.data).item())
    @torch.no_grad()
    def norm(self, x): return float(torch.linalg.vector_norm(x.data).item())

class EuclideanDual:
    def __init__(self, var=None): self.var = {} if var is None else var
    @torch.no_grad()
    def apply(self, x, y): return float(torch.sum(x.data * y.data).item())
    @torch.no_grad()
    def dual(self, x): return x
