from .Euclidean import L2Dual, L2Primal


class Problem:
    def __init__(self, obj_smooth, obj_nonsmooth, var=None):
        self.var = {} if var is None else dict(var)
        self.obj_smooth = obj_smooth
        self.obj_nonsmooth = obj_nonsmooth
        self.pvector = L2Primal(self.var)
        self.dvector = L2Dual(self.var)
