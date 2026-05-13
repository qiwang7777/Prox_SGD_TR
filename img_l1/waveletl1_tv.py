from semismooth_TR.tv_helper import grad2d, div2d, tv_value_isotropic,prox_tv_chambolle

import torch
import pywt
import numpy as np

class TVNonsmooth:
    def __init__(self, lam, key="img", prox_max_iter=2000, prox_tol=1e-10):
        self.lam = float(lam)
        self.key = key
        self.prox_max_iter = prox_max_iter
        self.prox_tol = prox_tol

    @torch.no_grad()
    def value(self, x):
        u = x.td[self.key]
        return float(self.lam * tv_value_isotropic(u).item())

    @torch.no_grad()
    def prox(self, x, t):
        out = x.copy()
        u = x.td[self.key]
        out.td[self.key] = prox_tv_chambolle(
            u,
            weight=t * self.lam,
            max_iter=self.prox_max_iter,
            tol=self.prox_tol
        )
        return out

class L1Nonsmooth:
    def __init__(self, lam, key="img"):
        self.lam = float(lam)
        self.key = key

    @torch.no_grad()
    def value(self, x):
        u = x.td[self.key]
        return float(self.lam * torch.sum(torch.abs(u)).item())

    @torch.no_grad()
    def prox(self, x, t):
        out = x.copy()
        u = x.td[self.key]
        thresh = t * self.lam
        out.td[self.key] = torch.sign(u) * torch.clamp(torch.abs(u) - thresh, min=0.0)
        return out
    
class WaveletL1Nonsmooth:
    """
    g(x) = lam * ||W x||_1,
    where W is a 2D wavelet transform applied to x.td[key].

    Prox for orthogonal wavelets:
        prox_{t lam ||W·||_1}(u) = W^{-1}( soft(Wu, t lam) )

    Notes:
    - Best to use an orthogonal wavelet such as 'db1', 'db2', 'haar', 'sym2', etc.
    - Reconstruction may be slightly larger than the original array due to boundary handling;
      we crop back to the original shape.
    """
    def __init__(self, lam, key="img", wavelet="db2", level=2, mode="periodization"):
        self.lam = float(lam)
        self.key = key
        self.wavelet = wavelet
        self.level = int(level)
        self.mode = mode

    @torch.no_grad()
    def _forward(self, u):
        u_np = u.detach().cpu().numpy()
        coeffs = pywt.wavedec2(
            u_np,
            wavelet=self.wavelet,
            level=self.level,
            mode=self.mode
        )
        arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        arr_t = torch.tensor(arr, dtype=u.dtype, device=u.device)
        return arr_t, coeff_slices, u.shape

    @torch.no_grad()
    def _inverse(self, arr_t, coeff_slices, out_shape, dtype, device):
        arr = arr_t.detach().cpu().numpy()
        coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format="wavedec2")
        u_rec = pywt.waverec2(coeffs, wavelet=self.wavelet, mode=self.mode)

        # Crop in case waverec2 returns a slightly larger array
        h, w = out_shape
        u_rec = u_rec[:h, :w]

        return torch.tensor(u_rec, dtype=dtype, device=device)

    @torch.no_grad()
    def value(self, x):
        u = x.td[self.key]
        w, _, _ = self._forward(u)
        return float(self.lam * torch.sum(torch.abs(w)).item())

    @torch.no_grad()
    def prox(self, x, t):
        out = x.copy()
        u = x.td[self.key]

        w, coeff_slices, out_shape = self._forward(u)

        thresh = t * self.lam
        w_soft = torch.sign(w) * torch.clamp(torch.abs(w) - thresh, min=0.0)

        out.td[self.key] = self._inverse(
            w_soft,
            coeff_slices,
            out_shape=out_shape,
            dtype=u.dtype,
            device=u.device
        )
        return out
    
class WaveletL1DetailNonsmooth:
    """
    g(x) = lam * sum |detail wavelet coefficients of x|

    The coarsest approximation coefficients are NOT penalized.
    Only detail coefficients are soft-thresholded in the prox.
    """
    def __init__(self, lam, key="img", wavelet="db2", level=2, mode="periodization"):
        self.lam = float(lam)
        self.key = key
        self.wavelet = wavelet
        self.level = int(level)
        self.mode = mode

    @torch.no_grad()
    def _wavedec(self, u):
        u_np = u.detach().cpu().numpy()
        coeffs = pywt.wavedec2(
            u_np,
            wavelet=self.wavelet,
            level=self.level,
            mode=self.mode
        )
        return coeffs, u.shape, u.dtype, u.device

    @torch.no_grad()
    def _waverec(self, coeffs, out_shape, dtype, device):
        u_rec = pywt.waverec2(coeffs, wavelet=self.wavelet, mode=self.mode)
        h, w = out_shape
        u_rec = u_rec[:h, :w]
        return torch.tensor(u_rec, dtype=dtype, device=device)

    @torch.no_grad()
    def value(self, x):
        u = x.td[self.key]
        coeffs, _, _, _ = self._wavedec(u)

        val = 0.0

        # coeffs[0] is the approximation block: do NOT penalize it
        for detail_level in coeffs[1:]:
            cH, cV, cD = detail_level
            val += np.sum(np.abs(cH))
            val += np.sum(np.abs(cV))
            val += np.sum(np.abs(cD))

        return float(self.lam * val)

    @torch.no_grad()
    def prox(self, x, t):
        out = x.copy()
        u = x.td[self.key]

        coeffs, out_shape, dtype, device = self._wavedec(u)
        thresh = t * self.lam

        new_coeffs = [coeffs[0]]  # keep approximation coefficients unchanged

        for detail_level in coeffs[1:]:
            cH, cV, cD = detail_level

            cH_t = np.sign(cH) * np.maximum(np.abs(cH) - thresh, 0.0)
            cV_t = np.sign(cV) * np.maximum(np.abs(cV) - thresh, 0.0)
            cD_t = np.sign(cD) * np.maximum(np.abs(cD) - thresh, 0.0)

            new_coeffs.append((cH_t, cV_t, cD_t))

        out.td[self.key] = self._waverec(
            new_coeffs,
            out_shape=out_shape,
            dtype=dtype,
            device=device
        )
        return out
