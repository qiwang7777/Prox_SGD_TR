import torch
import numpy as np

class DictEuclidean:
    """
    Euclidean inner product on TorchDictVector:
      <x,y> = sum_k sum_i x_k[i]*y_k[i]
    """
    def __init__(self, var=None):
        self.var = var

    @torch.no_grad()
    def dot(self, x, y):
        s = 0.0
        for k, vx in x.td.items():
            s += torch.sum(vx * y.td[k]).item()
        return float(s)

    @torch.no_grad()
    def apply(self, x, y):
        return self.dot(x, y)

    @torch.no_grad()
    def norm(self, x):
        return float(np.sqrt(max(self.dot(x, x), 0.0)))

    @torch.no_grad()
    def dual(self, x):
        return x


class L2TVPrimal(DictEuclidean):
    pass


class L2TVDual(DictEuclidean):
    pass

