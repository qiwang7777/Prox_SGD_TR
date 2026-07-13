import torch
import torch.nn.functional as F

from .Saturation import SaturationLeastSquares


class SmoothedSaturationLeastSquares(SaturationLeastSquares):
    """
    Saturation least-squares objective with an auxiliary smooth
    approximation of S(t) = clip(t, 0, 1).

    The inherited value(), gradient(), and hessVec() correspond to the
    original clipped objective.

    The methods value_smooth(), gradient_smooth(), and hessVec_smooth()
    correspond to the softplus-smoothed saturation mapping.
    """

    @staticmethod
    def S_smooth(z: torch.Tensor, mu: float) -> torch.Tensor:
        if mu <= 0.0:
            raise ValueError("mu must be positive.")

        mu_t = torch.as_tensor(mu, dtype=z.dtype, device=z.device)

        return mu_t * (
            F.softplus(z / mu_t)
            - F.softplus((z - 1.0) / mu_t)
        )

    @staticmethod
    def dS_smooth(z: torch.Tensor, mu: float) -> torch.Tensor:
        if mu <= 0.0:
            raise ValueError("mu must be positive.")

        mu_t = torch.as_tensor(mu, dtype=z.dtype, device=z.device)

        return (
            torch.sigmoid(z / mu_t)
            - torch.sigmoid((z - 1.0) / mu_t)
        )

    @staticmethod
    def ddS_smooth(z: torch.Tensor, mu: float) -> torch.Tensor:
        if mu <= 0.0:
            raise ValueError("mu must be positive.")

        mu_t = torch.as_tensor(mu, dtype=z.dtype, device=z.device)

        sig0 = torch.sigmoid(z / mu_t)
        sig1 = torch.sigmoid((z - 1.0) / mu_t)

        return (
            sig0 * (1.0 - sig0)
            - sig1 * (1.0 - sig1)
        ) / mu_t

    @torch.no_grad()
    def value_smooth(self, x, mu, tol=1e-12):
        u = x.td[self.key]
        Ax = self.A_apply(u)

        SAx = self.S_smooth(Ax, mu)
        residual = SAx - self.b

        value = 0.5 * torch.sum(residual.square())
        return float(value.item()), 0.0

    @torch.no_grad()
    def gradient_smooth(self, x, mu, tol=1e-12):
        u = x.td[self.key]
        Ax = self.A_apply(u)

        SAx = self.S_smooth(Ax, mu)
        dSAx = self.dS_smooth(Ax, mu)
        residual = SAx - self.b

        gradient = x.zero_like()
        gradient.td[self.key] = self.AT_apply(
            dSAx * residual
        )

        return gradient, 0.0

    @torch.no_grad()
    def hessVec_smooth(self, s, x, mu, tol=1e-12):
        """
        Exact Hessian-vector product for

            0.5 * ||S_mu(Ax) - b||^2.
        """
        u = x.td[self.key]
        v = s.td[self.key]

        Ax = self.A_apply(u)
        Av = self.A_apply(v)

        SAx = self.S_smooth(Ax, mu)
        dSAx = self.dS_smooth(Ax, mu)
        ddSAx = self.ddS_smooth(Ax, mu)

        residual = SAx - self.b

        coefficient = dSAx.square() + residual * ddSAx

        Hv = s.zero_like()
        Hv.td[self.key] = self.AT_apply(
            coefficient * Av
        )

        return Hv, 0.0
