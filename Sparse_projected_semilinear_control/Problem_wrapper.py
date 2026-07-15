from .Euclidean import EuclideanDual, EuclideanPrimal

class Problem:
    def __init__(self, obj_smooth, obj_nonsmooth, var=None):
        self.var = {} if var is None else dict(var)
        self.obj_smooth = obj_smooth
        self.obj_nonsmooth = obj_nonsmooth
        self.pvector = EuclideanPrimal(self.var)
        self.dvector = EuclideanDual(self.var)
