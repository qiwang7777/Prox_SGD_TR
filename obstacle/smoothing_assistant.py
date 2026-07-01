import torch
from collections import OrderedDict
from semismooth_TR.TorchVector import TorchDictVector


class SmoothingAssistant:
    """
    Generic AD-based smoothing wrapper.

    f_smooth(x, mu) must return a scalar torch.Tensor.
    Here x is a TorchDictVector.
    """

    def __init__(self, f_smooth):
        self.f_smooth = f_smooth

    def value(self, x, mu, tol=1e-12):
        val = self.f_smooth(x, mu)
        return float(val.detach().cpu().item()), 0.0

    def gradient(self, x, mu, tol=1e-12):
        td = OrderedDict()

        for k, v in x.td.items():
            td[k] = v.detach().clone().requires_grad_(True)

        x_ad = TorchDictVector(td)

        val = self.f_smooth(x_ad, mu)

        grads = torch.autograd.grad(
            val,
            list(td.values()),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )

        grad_td = OrderedDict()
        for (k, _), g in zip(td.items(), grads):
            grad_td[k] = g.detach().clone()

        return TorchDictVector(grad_td), 0.0

    def hessvec(self, x, v, mu, tol=1e-12):
        td = OrderedDict()

        for k, p in x.td.items():
            td[k] = p.detach().clone().requires_grad_(True)

        x_ad = TorchDictVector(td)

        val = self.f_smooth(x_ad, mu)

        grads = torch.autograd.grad(
            val,
            list(td.values()),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )

        gv = 0.0
        for (k, _), g in zip(td.items(), grads):
            gv = gv + torch.sum(g * v.td[k])

        hv = torch.autograd.grad(
            gv,
            list(td.values()),
            retain_graph=False,
            allow_unused=False,
        )

        hv_td = OrderedDict()
        for (k, _), h in zip(td.items(), hv):
            hv_td[k] = h.detach().clone()

        return TorchDictVector(hv_td), 0.0
