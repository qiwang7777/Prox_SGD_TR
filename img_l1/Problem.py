from.DictEuclidean import DictEuclidean, L2TVDual, L2TVPrimal

class Problem:
    """
    Container used by TR framework.

    Required fields used in your code:
      - obj_smooth: has update/value/gradient/hessVec
      - obj_nonsmooth: has value/prox
      - pvector: has dot/norm
      - dvector: has apply/dual
    """
    def __init__(self, obj_smooth, obj_nonsmooth, var=None):
        self.var = {} if var is None else dict(var)

        self.obj_smooth = obj_smooth
        self.obj_nonsmooth = obj_nonsmooth

        use_euclid = bool(self.var.get("useEuclidean", False))

        if use_euclid:
            self.pvector = DictEuclidean(self.var)
            self.dvector = DictEuclidean(self.var)
        else:
            self.pvector = L2TVPrimal(self.var)
            self.dvector = L2TVDual(self.var)