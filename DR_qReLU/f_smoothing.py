import torch
from collections import OrderedDict
from semismooth_TR.TorchVector import TorchDictVector


class SmoothingAssistant:
    def __init__(self, f_smooth):
        self.f_smooth = f_smooth

    def value(self, theta, mu, tol=1e-12):
        val = self.f_smooth(theta, mu)
        return float(val.detach().cpu().item()), 0.0

    def gradient(self, theta, mu, tol=1e-12):
        params = OrderedDict(
            (k, v.detach().clone().requires_grad_(True))
            for k, v in theta.td.items()
        )

        theta_ad = TorchDictVector(params)
        val = self.f_smooth(theta_ad, mu)

        grads = torch.autograd.grad(
            val,
            list(params.values()),
            create_graph=False,
            retain_graph=False,
        )

        grad_td = OrderedDict()
        for (name, _), g in zip(params.items(), grads):
            grad_td[name] = g.detach().clone()

        return TorchDictVector(grad_td), 0.0

    def hessvec(self, theta, v, mu, tol=1e-12):
        params = OrderedDict(
            (k, p.detach().clone().requires_grad_(True))
            for k, p in theta.td.items()
        )

        theta_ad = TorchDictVector(params)
        val = self.f_smooth(theta_ad, mu)

        grads = torch.autograd.grad(
            val,
            list(params.values()),
            create_graph=True,
            retain_graph=True,
        )

        gv = 0.0
        for (name, _), g in zip(params.items(), grads):
            gv = gv + torch.sum(g * v.td[name])

        hv = torch.autograd.grad(
            gv,
            list(params.values()),
            retain_graph=False,
        )

        hv_td = OrderedDict()
        for (name, _), h in zip(params.items(), hv):
            hv_td[name] = h.detach().clone()

        return TorchDictVector(hv_td), 0.0