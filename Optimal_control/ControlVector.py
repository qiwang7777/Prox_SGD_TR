import torch
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import math
from collections import deque
from PIL import Image

class ControlVector:
    """
    Optimization variable = control values on interior grid.
    Stored as tensor of shape (m,1).
    """
    def __init__(self, data):
        self.data = data.clone()

    def copy(self):
        return ControlVector(self.data.detach().clone())

    def clone(self):
        return ControlVector(self.data.clone())

    def zero_like(self):
        return ControlVector(torch.zeros_like(self.data))

    def randn_like(self):
        return ControlVector(torch.randn_like(self.data))

    def __add__(self, other):
        return ControlVector(self.data + other.data)

    def __sub__(self, other):
        return ControlVector(self.data - other.data)

    def __mul__(self, a: float):
        return ControlVector(a * self.data)

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

    def axpy(self, a: float, x: "ControlVector"):
        self.data = self.data + a * x.data
        return self

    def scal(self, a: float):
        self.data = a * self.data
        return self

    def dot(self, other) -> float:
        return float(torch.sum(self.data * other.data).item())

    def norm(self) -> float:
        return float(torch.sqrt(torch.sum(self.data * self.data)).item())

    def normalize_(self, eps: float = 1e-16):
        n = self.norm()
        if n > eps:
            self.data /= n
        return self