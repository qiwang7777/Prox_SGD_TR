import torch

class ControlVector:
    def __init__(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.data = data.clone()
    def copy(self): return ControlVector(self.data.detach().clone())
    def clone(self): return ControlVector(self.data.clone())
    def zero_like(self): return ControlVector(torch.zeros_like(self.data))
    def randn_like(self): return ControlVector(torch.randn_like(self.data))
    def __add__(self, other): return ControlVector(self.data + other.data)
    def __sub__(self, other): return ControlVector(self.data - other.data)
    def __mul__(self, a): return ControlVector(float(a) * self.data)
    def __rmul__(self, a): return self.__mul__(a)
    def __imul__(self, a): self.data = float(a) * self.data; return self
    def __iadd__(self, other): self.data += other.data; return self
    def __isub__(self, other): self.data -= other.data; return self
    def axpy(self, a, x): self.data += float(a) * x.data; return self
    def scal(self, a): self.data *= float(a); return self
    def dot(self, other): return float(torch.sum(self.data * other.data).item())
    def norm(self): return float(torch.linalg.vector_norm(self.data).item())
    def normalize_(self, eps=1e-16):
        nrm = self.norm()
        if nrm > eps: self.data /= nrm
        return self
