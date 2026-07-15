import torch


class GridVector:
    """
    Optimization variable on a uniform ngrid x ngrid grid.

    The data are stored as a tensor of shape (ngrid * ngrid, 1), matching
    the vector interface used by semismooth_TR.
    """

    def __init__(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.data = data.clone()

    def copy(self):
        return GridVector(self.data.detach().clone())

    def clone(self):
        return GridVector(self.data.clone())

    def zero_like(self):
        return GridVector(torch.zeros_like(self.data))

    def randn_like(self):
        return GridVector(torch.randn_like(self.data))

    def __add__(self, other):
        return GridVector(self.data + other.data)

    def __sub__(self, other):
        return GridVector(self.data - other.data)

    def __mul__(self, a: float):
        return GridVector(a * self.data)

    def __rmul__(self, a: float):
        return self.__mul__(a)

    def __imul__(self, a: float):
        self.data = a * self.data
        return self

    def __iadd__(self, other):
        self.data = self.data + other.data
        return self

    def __isub__(self, other):
        self.data = self.data - other.data
        return self

    def axpy(self, a: float, x: "GridVector"):
        self.data = self.data + a * x.data
        return self

    def scal(self, a: float):
        self.data = a * self.data
        return self

    def dot(self, other) -> float:
        return float(torch.sum(self.data * other.data).item())

    def norm(self) -> float:
        return float(torch.linalg.vector_norm(self.data).item())

    def normalize_(self, eps: float = 1e-16):
        nrm = self.norm()
        if nrm > eps:
            self.data /= nrm
        return self
