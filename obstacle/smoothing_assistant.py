import torch


class SmoothingAssistant:
    """
    Generic AD smoother.

    f_smooth(x, mu) must return a scalar torch Tensor.
    """

    def __init__(self, f_smooth):
        self.f_smooth = f_smooth

    def value(self, x, mu, tol=1e-12):
        val = self.f_smooth(x, mu)
        return float(val.detach().cpu().item()), 0.0

    def gradient(self, x, mu, tol=1e-12):
        x_ad = x.detach().clone().requires_grad_(True)
        val = self.f_smooth(x_ad, mu)

        grad = torch.autograd.grad(
            val,
            x_ad,
            create_graph=False,
            retain_graph=False,
        )[0]

        return grad.detach(), 0.0

    def hessvec(self, x, v, mu, tol=1e-12):
        x_ad = x.detach().clone().requires_grad_(True)
        val = self.f_smooth(x_ad, mu)

        grad = torch.autograd.grad(
            val,
            x_ad,
            create_graph=True,
            retain_graph=True,
        )[0]

        gv = torch.dot(grad.reshape(-1), v.reshape(-1))

        hv = torch.autograd.grad(
            gv,
            x_ad,
            retain_graph=False,
        )[0]

        return hv.detach(), 0.0
