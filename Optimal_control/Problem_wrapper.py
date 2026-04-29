from .Euclidean import L2TVDual,L2TVPrimal
class Problem:
    def __init__(self, obj_smooth, obj_nonsmooth, var=None):
        self.var = {} if var is None else dict(var)
        self.obj_smooth = obj_smooth
        self.obj_nonsmooth = obj_nonsmooth
        self.pvector = L2TVPrimal(self.var)
        self.dvector = L2TVDual(self.var)