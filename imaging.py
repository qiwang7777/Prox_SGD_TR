# ============================================================
# Full runnable script:
# TV-regularized image reconstruction with blur + clipping
#
# Problem:
#   min_x  0.5 * || S(Ax) - b ||^2 + lam * TV(x) # replace the TV(x) with ||Wx||_1 (wavelet basis)
#   S(t) = clamp(t, 0, 1)
#
# Uses TR framework and TorchDictVector structure.
# ============================================================

import math
import copy
import pywt
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import OrderedDict, deque
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
})

# ============================================================
# 1) TorchDictVector
# ============================================================

class TorchDictVector:
    """
    A lightweight vector wrapper:
    - stores parameters in a dict: .td[name] = tensor
    - supports +, -, scalar *, deep-ish copy/clone
    """
    def __init__(self, td=None):
        self.td = OrderedDict() if td is None else OrderedDict(td)

    def copy(self):
        return TorchDictVector({k: v.detach().clone() for k, v in self.td.items()})

    def clone(self):
        return TorchDictVector({k: v.clone() for k, v in self.td.items()})

    def zero_like(self):
        return TorchDictVector({k: torch.zeros_like(v) for k, v in self.td.items()})

    def __add__(self, other):
        out = TorchDictVector()
        for k in self.td.keys():
            out.td[k] = self.td[k] + other.td[k]
        return out

    def __sub__(self, other):
        out = TorchDictVector()
        for k in self.td.keys():
            out.td[k] = self.td[k] - other.td[k]
        return out

    def __mul__(self, a: float):
        out = TorchDictVector()
        for k, v in self.td.items():
            out.td[k] = a * v
        return out

    def __rmul__(self, a: float):
        return self.__mul__(a)

    def __iadd__(self, other):
        for k in self.td.keys():
            self.td[k] = self.td[k] + other.td[k]
        return self

    def __isub__(self, other):
        for k in self.td.keys():
            self.td[k] = self.td[k] - other.td[k]
        return self

    def randn_like(self):
        return TorchDictVector({k: torch.randn_like(v) for k, v in self.td.items()})

    def dot(self, other) -> float:
        s = 0.0
        for k, v in self.td.items():
            s += torch.sum(v * other.td[k]).item()
        return float(s)

    def axpy(self, a: float, x: "TorchDictVector"):
        for k in self.td.keys():
            self.td[k] = self.td[k] + a * x.td[k]
        return self

    def scal(self, a: float):
        for k in self.td.keys():
            self.td[k] = a * self.td[k]
        return self

    def norm(self) -> float:
        s = 0.0
        for v in self.td.values():
            s += torch.sum(v * v).item()
        return float(s ** 0.5)

    def normalize_(self, eps: float = 1e-16):
        n = self.norm()
        if n < eps:
            return self
        return self.scal(1.0 / n)


# ============================================================
# 2) Geometry / vector spaces
# ============================================================

class DictEuclidean:
    """
    Euclidean inner product on TorchDictVector:
      <x,y> = sum_k sum_i x_k[i]*y_k[i]
    """
    def __init__(self, var=None):
        self.var = var

    @torch.no_grad()
    def dot(self, x, y):
        s = 0.0
        for k, vx in x.td.items():
            s += torch.sum(vx * y.td[k]).item()
        return float(s)

    @torch.no_grad()
    def apply(self, x, y):
        return self.dot(x, y)

    @torch.no_grad()
    def norm(self, x):
        return float(np.sqrt(max(self.dot(x, x), 0.0)))

    @torch.no_grad()
    def dual(self, x):
        return x


class L2TVPrimal(DictEuclidean):
    pass


class L2TVDual(DictEuclidean):
    pass


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


# ============================================================
# 3) TV helpers
# ============================================================

def grad2d(u):
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)

    gx[:-1, :] = u[1:, :] - u[:-1, :]
    gy[:, :-1] = u[:, 1:] - u[:, :-1]
    return gx, gy


def div2d(px, py):
    out = torch.zeros_like(px)

    out[:-1, :] -= px[:-1, :]
    out[1:,  :] += px[:-1, :]

    out[:, :-1] -= py[:, :-1]
    out[:, 1: ] += py[:, :-1]

    return out


def tv_value_isotropic(u):
    gx, gy = grad2d(u)
    return torch.sum(torch.sqrt(gx * gx + gy * gy)) #+ 1e-16))


@torch.no_grad()
def prox_tv_chambolle(v, weight, max_iter=100, tol=1e-5):
    """
    Computes prox_{weight * TV}(v) for 2D isotropic TV:
        argmin_u 0.5||u-v||^2 + weight*TV(u)
    """
    if weight <= 0:
        return v.detach().clone()

    px = torch.zeros_like(v)
    py = torch.zeros_like(v)

    tau = 0.25

    for _ in range(max_iter):
        px_old = px.clone()
        py_old = py.clone()

        div_p = div2d(px, py)
        gx, gy = grad2d(div_p - v / weight)

        px = px + tau * gx
        py = py + tau * gy

        norm = torch.maximum(torch.ones_like(v), torch.sqrt(px * px + py * py))
        px = px / norm
        py = py / norm

        err = max((px - px_old).abs().max().item(), (py - py_old).abs().max().item())
        if err < tol:
            break

    return v - weight * div2d(px, py)


# ============================================================
# 4) Problem-specific nonsmooth part
# ============================================================

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

class WaveletL1DetailNonsmooth_L2:
    """
    g(x) = lam_detail * sum |detail wavelet coefficients of x|
         + 0.5 * lam_approx * ||approximation coefficients||_2^2

    The coarsest approximation coefficients are penalized weakly in L2.
    Detail coefficients are penalized in L1.

    Prox:
      - approximation block: shrink by quadratic prox
      - detail blocks: soft-threshold
    """
    def __init__(
        self,
        lam,
        key="img",
        wavelet="db2",
        level=2,
        mode="periodization",
        lam_approx=1e-4,
    ):
        self.lam = float(lam)                 # detail L1 weight
        self.key = key
        self.wavelet = wavelet
        self.level = int(level)
        self.mode = mode
        self.lam_approx = float(lam_approx)   # approximation L2 weight

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

        cA = coeffs[0]
        val = 0.5 * self.lam_approx * np.sum(cA * cA)

        for detail_level in coeffs[1:]:
            cH, cV, cD = detail_level
            val += self.lam * np.sum(np.abs(cH))
            val += self.lam * np.sum(np.abs(cV))
            val += self.lam * np.sum(np.abs(cD))

        return float(val)

    @torch.no_grad()
    def prox(self, x, t):
        out = x.copy()
        u = x.td[self.key]

        coeffs, out_shape, dtype, device = self._wavedec(u)

        thresh = t * self.lam

        # approximation block prox for 0.5*lam_approx*||cA||^2:
        # prox(v) = v / (1 + t*lam_approx)
        cA = coeffs[0]
        cA_new = cA / (1.0 + t * self.lam_approx)

        new_coeffs = [cA_new]

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
# ============================================================
# 5) Data generation
# ============================================================

def make_shepp_logan_data(n=128, sigma=2.0, kernel_size=3, noise_level=0.0, device="cpu"):
    x_true = shepp_logan_phantom()
    x_true = resize(x_true, (n, n), anti_aliasing=True)
    x_true = torch.tensor(x_true, dtype=torch.float64, device=device)

    ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=torch.float64)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    def A_apply(img):
        z = img.view(1, 1, n, n)
        out = F.conv2d(z, kernel, padding=kernel_size // 2)
        return out.view(n, n)

    def AT_apply(img):
        z = img.view(1, 1, n, n)
        out = F.conv2d(z, kernel, padding=kernel_size // 2)
        return out.view(n, n)

    def S(z):
        return torch.clamp(z, 0.0, 1.0)

    y_clean = A_apply(x_true)
    b = S(y_clean)
    if noise_level > 0:
        b = torch.clamp(b + noise_level * torch.randn_like(b), 0.0, 1.0)

    return x_true, b, A_apply, AT_apply, S


# ============================================================
# 6) Smooth part: 0.5 || clamp(Ax,0,1) - b ||^2
# ============================================================

class SaturationLeastSquares:
    """
    Smooth part:
        f(x) = 0.5 * || clamp(Ax, 0, 1) - b ||^2

    For optimization derivatives, this class can use a smoothed clamp near
    the kinks at 0 and 1, instead of the hard mask/interior suppression.

    Recommended usage:
        - value()      : true clipped objective
        - value_model(): smoothed local model
        - gradient()  : gradient of smoothed local model
        - hessVec()   : Gauss-Newton Hessian of smoothed local model

    The smoothed clamp is C^1 with a cubic transition over width mu_smooth
    near 0 and 1.

    If use_smoothed_kink=False, it falls back to the original masked model.
    """
    def __init__(
        self,
        b,
        A_apply,
        AT_apply,
        x_true=None,
        key="img",
        kink_tol=1e-8,
        use_reactivation=True,
        reactivation_only_when_all_saturated=True,
        mu_smooth=1e-2,
        use_smoothed_kink=True,
        use_full_hessian=False,
    ):
        self.b = b
        self.A_apply = A_apply
        self.AT_apply = AT_apply
        self.x_true = x_true
        self.key = key
        self.kink_tol = kink_tol
        self.use_reactivation = use_reactivation
        self.reactivation_only_when_all_saturated = reactivation_only_when_all_saturated

        self.mu_smooth = float(mu_smooth)
        self.use_smoothed_kink = bool(use_smoothed_kink)
        self.use_full_hessian = bool(use_full_hessian)

        # placeholders used elsewhere in your TR framework
        self.xy = None
        self.g = None
        self.weight = None
        self.V = None
        self.dV = None

        self.xy_full = None
        self.g_full = None
        self.weight_full = None
        self.V_full = None
        self.dV_full = None

        self._Ax = None
        self._SAx = None
        self._res = None
        self._mask_cache = None
        self._react_v = None
        self._react_mask = None
        self._using_reactivation = False

    @staticmethod
    def S(z):
        return torch.clamp(z, 0.0, 1.0)

    def _mask(self, z):
        return ((z > self.kink_tol) & (z < 1.0 - self.kink_tol)).to(z.dtype)

    def _reactivation_vector(self, z):
        v = torch.zeros_like(z)
        v = torch.where(z < 0.0, z, v)
        v = torch.where(z > 1.0, z - 1.0, v)
        return v

    def _reactivation_mask(self, z):
        return ((z < 0.0) | (z > 1.0)).to(z.dtype)

    def _should_use_reactivation(self, free_mask):
        if not self.use_reactivation:
            return False
        if self.reactivation_only_when_all_saturated:
            return bool(torch.count_nonzero(free_mask).item() == 0)
        return bool(torch.count_nonzero(free_mask).item() == 0)

    # ============================================================
    # Smoothed clamp pieces
    # ============================================================

    def _smoothstep01(self, t):
        # cubic smoothstep on [0,1]
        return 3.0 * t * t - 2.0 * t * t * t

    def _smoothstep01_prime(self, t):
        return 6.0 * t - 6.0 * t * t

    def _smoothstep01_second(self, t):
        return 6.0 - 12.0 * t

    def S_smooth(self, z, mu=None):
        """
        Smoothed clamp:
          - z <= -mu        -> 0
          - -mu < z < mu    -> smooth transition from 0 to identity
          - mu <= z <= 1-mu -> identity
          - 1-mu < z < 1+mu -> smooth transition from identity to 1
          - z >= 1+mu       -> 1
        """
        if mu is None:
            mu = self.mu_smooth

        out = torch.empty_like(z)

        r1 = z <= -mu
        r2 = (z > -mu) & (z < mu)
        r3 = (z >= mu) & (z <= 1.0 - mu)
        r4 = (z > 1.0 - mu) & (z < 1.0 + mu)
        r5 = z >= 1.0 + mu

        out[r1] = 0.0
        out[r3] = z[r3]
        out[r5] = 1.0

        # lower transition
        if torch.any(r2):
            t = (z[r2] + mu) / (2.0 * mu)  # maps [-mu, mu] -> [0,1]
            q = self._smoothstep01(t)
            out[r2] = z[r2] * q

        # upper transition
        if torch.any(r4):
            t = (z[r4] - (1.0 - mu)) / (2.0 * mu)  # maps [1-mu,1+mu] -> [0,1]
            q = self._smoothstep01(t)
            out[r4] = (1.0 - q) * z[r4] + q * 1.0

        return out

    def S_smooth_prime(self, z, mu=None):
        if mu is None:
            mu = self.mu_smooth

        out = torch.zeros_like(z)

        r2 = (z > -mu) & (z < mu)
        r3 = (z >= mu) & (z <= 1.0 - mu)
        r4 = (z > 1.0 - mu) & (z < 1.0 + mu)

        out[r3] = 1.0

        if torch.any(r2):
            t = (z[r2] + mu) / (2.0 * mu)
            q = self._smoothstep01(t)
            qp = self._smoothstep01_prime(t)
            dt_dz = 1.0 / (2.0 * mu)
            out[r2] = q + z[r2] * qp * dt_dz

        if torch.any(r4):
            t = (z[r4] - (1.0 - mu)) / (2.0 * mu)
            q = self._smoothstep01(t)
            qp = self._smoothstep01_prime(t)
            dt_dz = 1.0 / (2.0 * mu)
            out[r4] = (1.0 - q) + (1.0 - z[r4]) * qp * dt_dz

        return out

    def S_smooth_second(self, z, mu=None):
        if mu is None:
            mu = self.mu_smooth

        out = torch.zeros_like(z)

        r2 = (z > -mu) & (z < mu)
        r4 = (z > 1.0 - mu) & (z < 1.0 + mu)

        if torch.any(r2):
            t = (z[r2] + mu) / (2.0 * mu)
            qp = self._smoothstep01_prime(t)
            qpp = self._smoothstep01_second(t)
            dt_dz = 1.0 / (2.0 * mu)
            out[r2] = 2.0 * qp * dt_dz + z[r2] * qpp * (dt_dz ** 2)

        if torch.any(r4):
            t = (z[r4] - (1.0 - mu)) / (2.0 * mu)
            qp = self._smoothstep01_prime(t)
            qpp = self._smoothstep01_second(t)
            dt_dz = 1.0 / (2.0 * mu)
            out[r4] = -2.0 * qp * dt_dz + (1.0 - z[r4]) * qpp * (dt_dz ** 2)

        return out

    # ============================================================
    # State update
    # ============================================================

    def update(self, x, mode="accept"):
        u = x.td[self.key]
        self._Ax = self.A_apply(u)
        self._SAx = self.S(self._Ax)
        self._res = self._SAx - self.b
        self._mask_cache = self._mask(self._Ax)

        self._react_v = self._reactivation_vector(self._Ax)
        self._react_mask = self._reactivation_mask(self._Ax)
        self._using_reactivation = self._should_use_reactivation(self._mask_cache)

    # ============================================================
    # Objective values
    # ============================================================

    @torch.no_grad()
    #def value(self, x, tol=1e-12):
    #    """
    #    True clipped objective.
    #    """
    #    u = x.td[self.key]
    #    Ax = self.A_apply(u)
    #    SAx = self.S(Ax)
    #    res = SAx - self.b
    #    val = 0.5 * torch.sum(res * res)
    #    return float(val.item()), 0.0
    def value(self, x, tol=1e-12):
        u = x.td[self.key]
        Ax = self.A_apply(u)
        if self.use_smoothed_kink:
            SAx = self.S_smooth(Ax, self.mu_smooth)
        else:
            SAx = self.S(Ax)
        res = SAx - self.b
        val = 0.5 * torch.sum(res * res)
        return float(val.item()),0.0

    @torch.no_grad()
    def value_model(self, x, tol=1e-12):
        """
        Model value used by TR subproblem logic.
        Uses smoothed clamp when enabled.
        """
        u = x.td[self.key]
        Ax = self.A_apply(u)

        if self.use_smoothed_kink:
            SAx = self.S_smooth(Ax, self.mu_smooth)
            res = SAx - self.b
            val = 0.5 * torch.sum(res * res)
            return float(val.item()), 0.0

        mask = self._mask(Ax)
        if self._should_use_reactivation(mask):
            v = self._reactivation_vector(Ax)
            val = 0.5 * torch.sum(v * v)
            return float(val.item()), 0.0

        SAx = self.S(Ax)
        res = SAx - self.b
        val = 0.5 * torch.sum(res * res)
        return float(val.item()), 0.0

    # ============================================================
    # Gradient and Hessian-vector
    # ============================================================

    @torch.no_grad()
    def gradient(self, x, tol=1e-12):
        u = x.td[self.key]
        Ax = self.A_apply(u)
        g = x.zero_like()

        if self.use_smoothed_kink:
            Sz = self.S_smooth(Ax, self.mu_smooth)
            Sp = self.S_smooth_prime(Ax, self.mu_smooth)
            res = Sz - self.b
            g_img = self.AT_apply(Sp * res)
            g.td[self.key] = g_img
            return g, 0.0

        mask = self._mask(Ax)
        if self._should_use_reactivation(mask):
            v = self._reactivation_vector(Ax)
            g_img = self.AT_apply(v)
            g.td[self.key] = g_img
            return g, 0.0

        SAx = self.S(Ax)
        res = SAx - self.b
        g_img = self.AT_apply(mask * res)
        g.td[self.key] = g_img
        return g, 0.0

    @torch.no_grad()
    def hessVec(self, s, x, tol=1e-12):
        u = x.td[self.key]
        Ax = self.A_apply(u)
        As = self.A_apply(s.td[self.key])
        Hs = s.zero_like()

        if self.use_smoothed_kink:
            Sz = self.S_smooth(Ax, self.mu_smooth)
            Sp = self.S_smooth_prime(Ax, self.mu_smooth)

            if self.use_full_hessian:
                Spp = self.S_smooth_second(Ax, self.mu_smooth)
                res = Sz - self.b
                coeff = Sp * Sp + res * Spp
            else:
                # safer Gauss-Newton-type approximation
                coeff = Sp * Sp

            Hs_img = self.AT_apply(coeff * As)
            Hs.td[self.key] = Hs_img
            return Hs, 0.0

        mask = self._mask(Ax)
        if self._should_use_reactivation(mask):
            react_mask = self._reactivation_mask(Ax)
            Hs_img = self.AT_apply(react_mask * As)
            Hs.td[self.key] = Hs_img
            return Hs, 0.0

        Hs_img = self.AT_apply(mask * As)
        Hs.td[self.key] = Hs_img
        return Hs, 0.0

    # ============================================================
    # Diagnostics
    # ============================================================

    @torch.no_grad()
    def relative_L2_error(self, x):
        if self.x_true is None:
            return np.nan
        u = x.td[self.key]
        num = torch.norm(u - self.x_true)
        den = torch.norm(self.x_true) + 1e-16
        return float((num / den).item())

    @torch.no_grad()
    def diagnostics(self, x):
        u = x.td[self.key]
        Ax = self.A_apply(u)
        free = ((Ax > self.kink_tol) & (Ax < 1.0 - self.kink_tol)).sum().item()
        below = (Ax < 0.0).sum().item()
        above = (Ax > 1.0).sum().item()
        at_lower = torch.abs(Ax) <= self.kink_tol
        at_upper = torch.abs(Ax - 1.0) <= self.kink_tol
        kink = (at_lower | at_upper).sum().item()

        if self.use_smoothed_kink:
            near_lower = (torch.abs(Ax) < self.mu_smooth).sum().item()
            near_upper = (torch.abs(Ax - 1.0) < self.mu_smooth).sum().item()
        else:
            near_lower = 0
            near_upper = 0

        return {
            "free": int(free),
            "below_0": int(below),
            "above_1": int(above),
            "at_kink": int(kink),
            "near_lower_smoothing_band": int(near_lower),
            "near_upper_smoothing_band": int(near_upper),
            "using_smoothed_kink": bool(self.use_smoothed_kink),
            "using_reactivation": bool(self._should_use_reactivation(self._mask(Ax))),
        }


# ============================================================
# 7) Trust-region subroutines
# ============================================================

def trustregion_gcp2(x, val, grad, dgrad, phi, problem, params, cnt):
    params.setdefault('safeguard', np.sqrt(np.finfo(float).eps))
    params.setdefault('lam_min', 1e-12)
    params.setdefault('lam_max', 1e12)
    params.setdefault('t', 1)
    params.setdefault('t_gcp', params['t'])
    params.setdefault('gradTol', np.sqrt(np.finfo(float).eps))

    Hg, _ = problem.obj_smooth.hessVec(grad, x, params['gradTol'])
    gHg = problem.dvector.apply(Hg, grad)
    gg = problem.pvector.dot(grad, grad)

    if gHg > params['safeguard'] * gg:
        t0Tmp = gg / gHg
    else:
        t0Tmp = params['t'] / np.sqrt(gg)

    t0 = np.min([params['lam_max'], np.max([params['lam_min'], t0Tmp])])
    xc = problem.obj_nonsmooth.prox(x - t0 * dgrad, t0)
    cnt['nprox'] += 1

    s = xc - x
    snorm = problem.pvector.norm(s)

    Hs, _ = problem.obj_smooth.hessVec(s, x, params['gradTol'])
    sHs = problem.dvector.apply(Hs, s)
    gs = problem.pvector.dot(grad, s)

    phinew = problem.obj_nonsmooth.value(xc)
    cnt['nobj2'] += 1

    alpha = 1
    if snorm >= (1 - params['safeguard']) * params['delta']:
        alpha = np.minimum(1, params['delta'] / snorm)

    if sHs > params['safeguard']:
        alpha = np.minimum(alpha, -(gs + phinew - phi) / sHs)
        #alpha_model = -(gs+phinew-phi)/sHs
        #if np.isfinite(alpha_model) and alpha_model>0.0:
        #    alpha = np.minimum(alpha,alpha_model)

    if alpha != 1:
        s *= alpha
        snorm *= alpha
        gs *= alpha
        Hs *= alpha
        sHs *= alpha ** 2
        xc = x + s
        phinew = problem.obj_nonsmooth.value(xc)
        cnt['nobj2'] += 1

    valnew = val + gs + 0.5 * sHs
    pRed = (val + phi) - (valnew + phinew)
    params['t_gcp'] = t0
    return s, snorm, pRed, phinew, Hs, cnt, params


def trustregion_gcp1(x, val, dgrad, phi, problem, params, cnt):
    params.setdefault("safeguard", np.sqrt(np.finfo(float).eps))
    params.setdefault("lam_min", 1e-12)
    params.setdefault("lam_max", 1e12)
    params.setdefault("t", 1.0)
    params.setdefault("t_gcp", params["t"])
    params.setdefault("gradTol", np.sqrt(np.finfo(float).eps))

    Hg, _ = problem.obj_smooth.hessVec(dgrad, x, params["gradTol"])
    cnt["nhess"] = cnt.get("nhess", 0) + 1

    gHg = problem.dvector.apply(Hg, dgrad)
    gg = problem.pvector.dot(dgrad, dgrad)

    if gHg > params["safeguard"] * gg:
        t0Tmp = gg / gHg
    else:
        t0Tmp = params["t"] / np.sqrt(gg)

    t0 = np.min([params["lam_max"], np.max([params["lam_min"], t0Tmp])])

    xc = problem.obj_nonsmooth.prox(x - t0 * dgrad, t0)
    cnt["nprox"] = cnt.get("nprox", 0) + 1

    s = xc - x
    snorm = problem.pvector.norm(s)

    Hs, _ = problem.obj_smooth.hessVec(s, x, params["gradTol"])
    cnt["nhess"] = cnt.get("nhess", 0) + 1

    sHs = problem.dvector.apply(Hs, s)
    gs = problem.pvector.dot(dgrad, s)

    phinew = problem.obj_nonsmooth.value(xc)
    cnt["nobj2"] = cnt.get("nobj2", 0) + 1

    alpha = 1.0
    if snorm >= (1 - params["safeguard"]) * params["delta"]:
        alpha = np.minimum(1.0, params["delta"] / snorm)

    if sHs > params["safeguard"]:
        alpha = np.minimum(alpha, -(gs + phinew - phi) / sHs)

    if alpha != 1.0:
        s *= alpha
        snorm *= alpha
        gs *= alpha
        Hs *= alpha
        sHs *= alpha ** 2
        xc = x + s
        phinew = problem.obj_nonsmooth.value(xc)
        cnt["nobj2"] = cnt.get("nobj2", 0) + 1

    valnew = val + gs + 0.5 * sHs
    pRed = (val + phi) - (valnew + phinew)
    params["t_gcp"] = t0

    return s, snorm, pRed, phinew, Hs, cnt, params


def dbls(nval, y, s, tmax, lambda_, kappa, gs, maxit, problem, cnt):
    tol = math.sqrt(np.finfo(float).eps)
    tol0 = 1e4 * tol
    eps0 = 1e2 * np.finfo(float).eps
    eps1 = 1e-2 * tol
    lam = 0.5 * (3 - math.sqrt(5))
    mu = 1e-4
    nold = nval

    tL = 0.0
    pL = 0.0
    nL = nold
    tR = tmax

    pwa = y + tR * s
    nR = problem.obj_nonsmooth.value(pwa)
    cnt["nobj2"] += 1

    if np.isinf(nR):
        t0 = lambda_
        pwa = y + t0 * s
        n0 = problem.obj_nonsmooth.value(pwa)
        cnt["nobj2"] += 1
        p0 = t0 * (0.5 * t0 * kappa + gs) + n0 - nold

        if p0 >= 0:
            tR = t0
            nR = n0
            pR = p0
        else:
            t2 = tR
            tolBS = tol * (tR - lambda_)

            while t2 > t0 + tolBS:
                t1 = 0.5 * (t0 + t2)
                pwa = y + t1 * s
                n1 = problem.obj_nonsmooth.value(pwa)
                cnt["nobj2"] += 1

                if np.isinf(n1):
                    t2 = t1
                else:
                    tp = t0
                    np0 = n0
                    pp = p0
                    t0 = t1
                    n0 = n1
                    p0 = t0 * (0.5 * t0 * kappa + gs) + n0 - nold

                    if p0 >= pp:
                        tL = tp
                        nL = np0
                        pL = pp
                        break

            tR = t0
            nR = n0
            pR = p0
    else:
        pR = tR * (0.5 * tR * kappa + gs) + nR - nold

    t = tR
    if kappa > 0:
        t = min(tR, -(((nR - nold) / (tR - tL)) + gs) / kappa)

    useOpT = True
    if t <= tL:
        t = tL + lam * (tR - tL)
        useOpT = False

    pwa = y + t * s
    nt = problem.obj_nonsmooth.value(pwa)
    cnt["nobj2"] += 1

    Qt = t * gs + nt - nold
    pt = 0.5 * kappa * t**2 + Qt

    if useOpT and nt == nold and nR == nold:
        alpha = t
        nval = nold
        return alpha, nval, cnt

    if pt >= max(pL, pR):
        alpha = tR
        nval = nR
        return alpha, nval, cnt

    v = tL
    pv = pL
    w = tR
    pw = pR
    d = 0.0
    e = 0.0

    tm = 0.5 * (tL + tR)
    tol1 = tol0 * abs(t) + eps1
    tol2 = 2 * tol1

    for _ in range(maxit):
        dL = tL - t
        dR = tR - t

        if abs(e) > tol1:
            r = (t - w) * (pt - pv)
            q = (t - v) * (pt - pw)
            p = (t - v) * q - (t - w) * r
            q = 2 * (q - r)

            if q > 0:
                p = -p
            q = abs(q)

            etmp = e
            e = d

            if abs(p) >= abs(0.5 * q * etmp) or p <= q * dL or p >= q * dR:
                e = dR
                if t > tm:
                    e = dL
                d = lam * e
            else:
                d = p / q
                u = t + d
                if (u - tL < tol2) or (tR - u < tol2):
                    d = tol1
                    if tm < t:
                        d = -tol1
        else:
            e = dR
            if t > tm:
                e = dL
            d = lam * e

        u = t + d
        if abs(d) < tol1:
            u = t + tol1
            if d < 0:
                u = t - tol1

        pwa = y + u * s
        nu = problem.obj_nonsmooth.value(pwa)
        cnt["nobj2"] += 1

        Qu = u * gs + nu - nold
        pu = 0.5 * kappa * u**2 + Qu

        if pu <= pt:
            if u >= t:
                tL = t
            else:
                tR = t

            v = w
            w = t
            t = u
            pv = pw
            pw = pt
            pt = pu
            nt = nu
            Qt = Qu
        else:
            if u < t:
                tL = u
            else:
                tR = u

            if pu <= pw or w == t:
                v = w
                w = u
                pv = pw
                pw = pu
            elif pu <= pv or v == t or v == w:
                v = u
                pv = pu

        tm = 0.5 * (tL + tR)
        tol1 = tol0 * abs(t) + eps1
        tol2 = 2 * tol1
        tol3 = eps0 * max(abs(Qt), 1.0)

        if pt <= (mu * min(0.0, Qt) + tol3) and abs(t - tm) <= (tol2 - 0.5 * (tR - tL)):
            break

    alpha = t
    nval = nt
    return alpha, nval, cnt


def trustregion_step_NCG(x, val, dgrad, phi, problem, params, cnt):
    params.setdefault("debug", False)
    params.setdefault("gradTol", np.sqrt(np.finfo(float).eps))
    params.setdefault("t", 1.0)
    params.setdefault("safeguard", np.sqrt(np.finfo(float).eps))
    params.setdefault("maxitsp", 15)
    params.setdefault("atol", 1e-4)
    params.setdefault("rtol", 1e-2)
    params.setdefault("spexp", 2)
    params.setdefault("maxitdbls", 5)
    params.setdefault("ncg_type", 4)
    params.setdefault("eta", 1e-2)
    params.setdefault("descPar", 0.2)
    params.setdefault("lam_min", 1e-12)
    params.setdefault("lam_max", 1e12)

    del2 = params["delta"] ** 2
    snorm0 = 0.0
    ss0 = 0.0
    y = copy.deepcopy(x)
    gmod = copy.deepcopy(dgrad)
    valold = val
    phiold = phi

    if params.get("useGCP", False):
        sc, snormc, _, _, _, cnt, params = trustregion_gcp1(
            x, val, dgrad, phi, problem, params, cnt
        )
        lambda_ = params["t"]
    else:
        Hg, _ = problem.obj_smooth.hessVec(dgrad, x, params["gradTol"])
        cnt["nhess"] = cnt.get("nhess", 0) + 1

        gHg = problem.dvector.apply(Hg, dgrad)
        gg = problem.pvector.dot(dgrad, dgrad)

        if gHg > params["safeguard"] * gg:
            lambdaTmp = gg / gHg
        else:
            lambdaTmp = params["t"] / np.sqrt(gg)

        lambda_ = np.max([params["lam_min"], np.min([params["lam_max"], lambdaTmp])])

        sc = problem.obj_nonsmooth.prox(y - lambda_ * gmod, lambda_) - y
        cnt["nprox"] = cnt.get("nprox", 0) + 1
        snormc = problem.pvector.norm(sc)

    lam1 = lambda_
    dx = (1.0 / lambda_) * sc
    s = copy.deepcopy(dx)
    gs = problem.pvector.dot(gmod, s)
    snorm = snormc / lambda_
    gnorm = snorm
    gtol = np.min([params["rtol"] * gnorm ** params["spexp"], params["atol"]])

    iflag = 1
    iter_count = 0
    phinew = phiold

    for iter0 in range(1, params["maxitsp"] + 1):
        Hs, _ = problem.obj_smooth.hessVec(s, x, params["gradTol"])
        cnt["nhess"] = cnt.get("nhess", 0) + 1
        sHs = problem.dvector.apply(Hs, s)

        ds = problem.pvector.dot(s, y - x)
        ss = snorm ** 2
        alphaMax = (-ds + np.sqrt(ds**2 + ss * (del2 - ss0))) / ss

        alpha, phiold, cnt = dbls(
            phiold, y, s, alphaMax, lam1, sHs, gs,
            params["maxitdbls"], problem, cnt
        )

        y = y + alpha * s
        gmod = gmod + alpha * problem.dvector.dual(Hs)
        valold = valold + alpha * (gs + 0.5 * alpha * sHs)

        ss0 = alpha**2 * ss + 2 * alpha * ds + ss0
        snorm0 = np.sqrt(ss0)

        if snorm0 >= (1 - params["safeguard"]) * params["delta"]:
            iflag = 2
            iter_count = iter0
            break

        if sHs > params["safeguard"] * ss:
            lambdaTmp = ss / sHs
        else:
            lambdaTmp = params["t"] / problem.pvector.norm(gmod)

        lambda_ = np.max([params["lam_min"], np.min([params["lam_max"], lambdaTmp])])

        dx0 = copy.deepcopy(dx)
        gnorm0 = gnorm
        dx = (1.0 / lambda_) * (problem.obj_nonsmooth.prox(y - lambda_ * gmod, lambda_) - y)
        cnt["nprox"] = cnt.get("nprox", 0) + 1

        gnorm = problem.pvector.norm(dx)

        if gnorm <= gtol:
            iflag = 0
            iter_count = iter0
            break

        gnorm2 = gnorm ** 2

        if params["ncg_type"] == 0:
            beta = gnorm2 / (gnorm0 ** 2)
        elif params["ncg_type"] == 1:
            d = dx0 - dx
            beta = max(0.0, -problem.pvector.dot(d, dx) / (gnorm0 ** 2))
        elif params["ncg_type"] == 2:
            d = dx0 - dx
            sd = problem.pvector.dot(s, d)
            gg = problem.pvector.dot(dx, dx0)
            eta = -1.0 / (problem.pvector.norm(s) * min(params["eta"], gnorm0))
            beta = max(
                eta,
                (gnorm2 - gg
                 - 2 * problem.pvector.dot(s, dx) * (gnorm2 - 2 * gg + gnorm0**2) / sd) / sd
            )
        elif params["ncg_type"] == 3:
            d = dx0 - dx
            beta = max(0.0, -problem.pvector.dot(d, dx) / problem.pvector.dot(s, d))
        elif params["ncg_type"] == 4:
            d = dx0 - dx
            beta = max(0.0, gnorm2 / problem.pvector.dot(s, d))
        elif params["ncg_type"] == 5:
            d = dx0 - dx
            beta = min(gnorm2, max(-gnorm2, -problem.pvector.dot(d, dx))) / (gnorm0 ** 2)
        else:
            d = dx0 - dx
            beta = max(
                0.0,
                min(-problem.pvector.dot(d, dx), gnorm2) / problem.pvector.dot(s, d)
            )

        reset = True
        if beta != 0 and np.isfinite(beta):
            s0 = dx + beta * s
            gs0 = problem.pvector.dot(gmod, s0)
            phi_trial = problem.obj_nonsmooth.value(y + s0)
            cnt["nobj2"] = cnt.get("nobj2", 0) + 1

            if (gs0 + phi_trial - phiold) <= -(1 - params["descPar"]) * gnorm2:
                s = s0
                gs = gs0
                lam1 = 1.0
                reset = False

        if reset:
            beta = 0.0
            s = copy.deepcopy(dx)
            lam1 = lambda_
            gs = problem.pvector.dot(gmod, s)

        snorm = problem.pvector.norm(s)

        if params["debug"]:
            print(f"Computed Alpha:      {alpha: .6e}")
            print(f"Computed Beta:       {beta: .6e}")
            print(f"Reset:               {int(reset)}")
            print(f"Iflag:               {iflag}")
            print(f"Model Value:         {valold + phiold: .6e}")

        iter_count = iter0

    if iter_count == 0 and params["maxitsp"] > 0:
        iter_count = iter0

    s = y - x
    snorm = problem.pvector.norm(s)
    phinew = phiold
    pRed = (val + phi) - (valold + phinew)

    if params["debug"]:
        print(f"Iflag:               {iflag}")
        print(f"Initial Model Value: {val + phi: .6e}")
        print(f"Final Model Value:   {valold + phiold: .6e}")

    return s, snorm, pRed, phinew, iflag, iter_count, cnt, params



def trustregion_step_SPG2(x, val, grad, dgrad, phi, problem, params, cnt):
    params.setdefault('maxitsp', 50)
    params.setdefault('lam_min', 1e-12)
    params.setdefault('lam_max', 1e12)
    params.setdefault('t', 1)
    params.setdefault('gradTol', np.sqrt(np.finfo(float).eps))
    params.setdefault('safeguard', np.sqrt(np.finfo(float).eps))
    params.setdefault('atol', 1e-4)
    params.setdefault('rtol', 1e-2)
    params.setdefault('spexp', 2)

    x0 = copy.deepcopy(x)
    g0_primal = copy.deepcopy(grad)
    snorm = 0

    valold = val
    phiold = phi
    valnew = valold
    phinew = phiold

    sc, snormc, pRed, _, _, cnt, params = trustregion_gcp2(x, val, grad, dgrad, phi, problem, params, cnt)

    t0 = params['t']
    s = copy.deepcopy(sc)
    x1 = x0 + s
    gnorm = snormc
    gtol = np.min([params['atol'], params['rtol'] * (gnorm / t0) ** params['spexp']])

    iter = 0
    iflag = 1

    for iter0 in range(1, params['maxitsp'] + 1):
        alphamax = 1
        snorm0 = snorm
        snorm = problem.pvector.norm(x1 - x)

        if snorm >= (1 - params['safeguard']) * params['delta']:
            ds = problem.pvector.dot(s, x0 - x)
            dd = gnorm ** 2
            alphamax = np.minimum(1, (-ds + np.sqrt(ds ** 2 + dd * (params['delta'] ** 2 - snorm0 ** 2))) / dd)

        Hs, _ = problem.obj_smooth.hessVec(s, x, params['gradTol'])
        sHs = problem.dvector.apply(Hs, s)
        g0s = problem.pvector.dot(g0_primal, s)
        phinew = problem.obj_nonsmooth.value(x1)
        alpha0 = -(g0s + phinew - phiold) / sHs

        if (not np.isfinite(sHs)) or (sHs <= 1e-14):
            alpha = alphamax
        else:
            alpha0 = -(g0s+phinew-phiold)/sHs
            if alpha0 <= 0:
                alpha = alphamax
            else:
                alpha = min(alphamax,alpha0)
            #alpha = max(0.0, np.minimum(alphamax, alpha0))

        if alpha == 1:
            x0 = x1
            g0_primal += problem.dvector.dual(Hs)
            valnew = valold + g0s + 0.5 * sHs
        else:
            x0 += alpha * s
            g0_primal += alpha * problem.dvector.dual(Hs)
            valnew = valold + alpha * g0s + 0.5 * alpha ** 2 * sHs
            phinew = problem.obj_nonsmooth.value(x0)
            snorm = problem.pvector.norm(x0 - x)

        valold = valnew
        phiold = phinew

        if snorm >= (1 - params['safeguard']) * params['delta']:
            iflag = 2
            break

        if sHs <= params['safeguard']:
            lambdaTmp = params['t'] / problem.pvector.norm(g0_primal)
        else:
            lambdaTmp = gnorm ** 2 / sHs
        
        
        t0 = np.max([params['lam_min'], np.min([params['lam_max'], lambdaTmp])])
        v = x0 - t0 * g0_primal
        x1 = problem.obj_nonsmooth.prox(x0 - t0 * g0_primal, t0)
        cnt['nprox'] += 1
        s = x1 - x0

        gnorm = problem.pvector.norm(s)
        if gnorm / t0 <= gtol:
            iflag = 0
            break

    s = x0 - x
    pRed = (val + phi) - (valnew + phinew)
    if iter0 > iter:
        iter = iter0
    return s, snorm, pRed, phinew, iflag, iter, cnt, params


# ============================================================
# 8) Gradient computation
# ============================================================

def compute_gradient(x, problem, params, cnt):
    if params['useInexactGrad']:
        scale0 = params['scaleGradTol']
        gtol = min(params['maxGradTol'], scale0 * params['delta'])
        gerr = gtol + 1
        while gerr > gtol:
            grad, gerr = problem.obj_smooth.gradient(x, gtol)
            gerr = 0
            cnt['ngrad'] += 1
            dgrad = problem.dvector.dual(grad)
            pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
            cnt['nprox'] += 1
            gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']
            gtol = min(params['maxGradTol'], scale0 * min(gnorm, params['delta']))
    else:
        gtol = 1e-12
        grad, gerr = problem.obj_smooth.gradient(x, gtol)
        cnt['ngrad'] += 1
        dgrad = problem.dvector.dual(grad)
        pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
        cnt['nprox'] += 1
        gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']

    params['gradTol'] = gtol
    cnt['graderr'].append(gerr)
    cnt['gradtol'].append(gtol)
    return grad, dgrad, gnorm, cnt


# ============================================================
# 9) Main trust-region driver
# ============================================================

def trustregion(x0, Deltai, problem, params):
    start_time = time.time()

    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 500)
    params.setdefault('gtol', 1e-3)
    params.setdefault('stol', 1e-9)
    params.setdefault('ocScale', 1.0)
    params.setdefault('atol', 1e-4)
    params.setdefault('rtol', 1e-2)
    params.setdefault('spexp', 2)
    params.setdefault('eta1', 1e-4)
    params.setdefault('eta2', 0.5)
    params.setdefault('gamma1', 0.25)
    params.setdefault('gamma2', 1.5)
    params.setdefault('delta', Deltai)
    params.setdefault('deltamin', 1e-16)
    params.setdefault('deltamax', 100.0)
    params.setdefault('reltol', False)

    params.setdefault('delta_stop', 1e-7)
    params.setdefault('stol_abs', 1e-9)
    params.setdefault('stag_window', 10)
    params.setdefault('ftol_rel', 1e-6)
    params.setdefault('max_reject', 15)
    params.setdefault("nonmono_M", 10)

    params.setdefault("pred_abs_tol", 1e-11)
    params.setdefault("pred_rel_tol", 1e-11)
    params.setdefault("pred_small_max", 5)

    params.setdefault("useInexactGrad", False)
    params.setdefault("auto_inexact_grad", False)
    params.setdefault("scaleGradTol", 1e-2)
    params.setdefault("maxGradTol", 1e-3)
    params.setdefault("kink_tau", 5e-3)
    params.setdefault("grad_match_tol", 0.49)
    params.setdefault("grad_match_q", 0.90)

    params.setdefault("mu_smooth", 1e-4)

    cnt = {
        'AlgType': f"TR-{params.get('spsolver','SPG2')}",
        'iter': 0,
        'nobj1': 0,
        'ngrad': 0,
        'nobj2': 0,
        'nprox': 0,
        'timetotal': 0.0,
        'objhist': [],
        'obj1hist': [],
        'obj2hist': [],
        'gnormhist': [],
        'snormhist': [],
        'deltahist': [],
        'nobj1hist': [],
        'nobj2hist': [],
        'ngradhist': [],
        'nproxhist': [],
        'timehist': [],
        'valerr': [],
        'valtol': [],
        'graderr': [],
        'gradtol': []
    }

    obj = problem.obj_smooth

    if params['initProx']:
        x = problem.obj_nonsmooth.prox(x0, 1.0)
        cnt['nprox'] += 1
    else:
        x = x0.copy()
        
    best_x = x.copy()
    best_relL2 = obj.relative_L2_error(x) if obj.x_true is not None else np.inf
    cnt["best_relL2"] = best_relL2
    cnt["best_iter"] = 0

    obj.update(x, "init")

    obj.xy_full = obj.xy
    if hasattr(obj, "g"):
        obj.g_full = obj.g
    obj.weight_full = getattr(obj, "weight", None)
    if hasattr(obj, "V"):
        obj.V_full = obj.V
    if hasattr(obj, "dV"):
        obj.dV_full = obj.dV

    rej_count = 0
    small_pred_count = 0

    val_true, _ = obj.value(x, 1e-12)
    cnt['nobj1'] += 1

    if hasattr(obj, "value_model"):
        val_model, _ = obj.value_model(x, 1e-12)
        cnt['nobj1'] += 1
    else:
        val_model = val_true

    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)

    phi = problem.obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    Facc = [val_true + phi]
    Fhist = deque(maxlen=params["nonmono_M"])
    Fhist.append(val_true + phi)

    print(f"TR method using {params.get('spsolver','SPG2')} Subproblem Solver")
    print("  iter    value         gnorm        del          snorm        nobjs    ngrad    nobjn    nprox     iterSP    flagSP")
    print(f"{0:4d}    {val_true+phi:12.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    ---      "
          f"{cnt['nobj1']:6d}    {cnt['ngrad']:6d}    {cnt['nobj2']:6d}    {cnt['nprox']:6d}    ---    ---")

    cnt['objhist'].append(val_true + phi)
    cnt['obj1hist'].append(val_true)
    cnt['obj2hist'].append(phi)
    cnt['gnormhist'].append(gnorm)
    cnt['snormhist'].append(np.nan)
    cnt['deltahist'].append(params['delta'])
    cnt['nobj1hist'].append(cnt['nobj1'])
    cnt['nobj2hist'].append(cnt['nobj2'])
    cnt['ngradhist'].append(cnt['ngrad'])
    cnt['nproxhist'].append(cnt['nprox'])
    cnt['timehist'].append(np.nan)

    gtol = params['gtol']
    stol = params['stol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm
        stol = params['stol'] * gnorm

    for i in range(1, params['maxit'] + 1):
        params['tolsp'] = min(params['atol'], params['rtol'] * (gnorm ** params['spexp']))
        grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)

        if hasattr(obj, "value_model"):
            val_model, _ = obj.value_model(x, 1e-12)
        else:
            val_model = val_true
        cnt['nobj1'] += 1

        if params.get('spsolver', 'NCG').upper() == 'SPG2':
            s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
                x, val_model, grad, dgrad, phi, problem, params, cnt
            )
        else:
            s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_NCG(
                x, val_model, dgrad, phi, problem, params, cnt
            )

        pRed = float(pRed)
        pred_floor = max(
            params["pred_abs_tol"],
            params["pred_rel_tol"] * max(1.0, abs(val_model + phi))
        )

        if pRed <= pred_floor:
            small_pred_count += 1
        else:
            small_pred_count = 0

        if (small_pred_count >= params["pred_small_max"]) and (params["delta"] <= 10.0 * params["delta_stop"]):
            cnt['iter'] = i
            cnt['timetotal'] = time.time() - start_time
            cnt['iflag'] = 6
            print("Optimization terminated because predicted reduction is tiny repeatedly.")
            print(f"Total time: {cnt['timetotal']:8.6e} seconds")
            return x, cnt

        xnew = x + s

        valnew_true, _ = obj.value(xnew, 1e-12)
        cnt['nobj1'] += 1
        phinew_true = problem.obj_nonsmooth.value(xnew)
        cnt['nobj2'] += 1

        aRed = (val_true + phi) - (valnew_true + phinew_true)

        rho = -np.inf if pRed <= 0.0 else float(aRed) / pRed
        Fref = max(Fhist)
        accept_nm = (valnew_true + phinew_true) <= (Fref - 1e-12)
        accept = (rho >= params['eta1']) and accept_nm

        print("debug:",
              "aRed=", float(aRed),
              "pRed=", float(pRed),
              "rho=", float(rho))

        if not accept:
            params['delta'] = max(params['deltamin'], params['gamma1'] * params['delta'])
            obj.update(x, 'reject')
            rej_count += 1
        else:
            x = xnew
            phi = phinew_true
            val_true = valnew_true
            rej_count = 0

            obj.update(x, 'accept')
            print(problem.obj_smooth.diagnostics(x))

            Facc.append(val_true + phi)
            Fhist.append(val_true + phi)

            relL2 = obj.relative_L2_error(x)
            print("relative L2 error =", relL2)
            
            if relL2 < best_relL2:
                best_relL2 = relL2
                best_x = x.copy()
                cnt["best_relL2"] = best_relL2
                cnt["best_iter"] = i

            if rho > params['eta2']:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])

        if i % params['outFreq'] == 0:
            print(f"{i:4d}    {val_true + phi:12.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}    "
                  f"{cnt['nobj1']:6d}    {cnt['ngrad']:6d}    {cnt['nobj2']:6d}    {cnt['nprox']:6d}    "
                  f"{iter_count:4d}    {iflag:1d}")

        cnt['objhist'].append(val_true + phi)
        cnt['obj1hist'].append(val_true)
        cnt['obj2hist'].append(phi)
        cnt['gnormhist'].append(gnorm)
        cnt['snormhist'].append(snorm)
        cnt['deltahist'].append(params['delta'])
        cnt['nobj1hist'].append(cnt['nobj1'])
        cnt['nobj2hist'].append(cnt['nobj2'])
        cnt['ngradhist'].append(cnt['ngrad'])
        cnt['nproxhist'].append(cnt['nprox'])
        cnt['timehist'].append(time.time() - start_time)

        delta_stop = params["delta_stop"]
        stol_abs = params["stol_abs"]
        K = params["stag_window"]
        ftol_rel = params["ftol_rel"]
        max_reject = params["max_reject"]

        stop_grad = (gnorm <= gtol)
        stop_step = (snorm < stol_abs) and (params["delta"] <= delta_stop)
        stop_stuck = (params["delta"] <= 10 * delta_stop and rej_count >= max_reject)

        stop_stag = False
        if len(Facc) >= K + 1:
            Fold = Facc[-(K + 1)]
            Fnew = Facc[-1]
            rel_change = abs(Fold - Fnew) / max(1.0, abs(Fnew))
            stop_stag = (rel_change < ftol_rel)

        stop_maxit = (i >= params["maxit"])

        if stop_grad or stop_step or stop_stag or stop_stuck or stop_maxit:
            if stop_grad:
                flag = 0
                reason = "gradient tolerance met"
            elif stop_step:
                flag = 2
                reason = "step small and TR radius collapsed"
            elif stop_stag:
                flag = 3
                reason = "objective stagnation"
            elif stop_stuck:
                flag = 4
                reason = "trust region stuck (delta small + rejections)"
            else:
                flag = 1
                reason = "maximum iterations reached"

            cnt['iter'] = i
            cnt['timetotal'] = time.time() - start_time
            cnt['iflag'] = flag

            print("Optimization terminated because", reason)
            print(f"Total time: {cnt['timetotal']:8.6e} seconds")
            return x, cnt, best_x

    cnt['iter'] = params['maxit']
    cnt['timetotal'] = time.time() - start_time
    cnt['iflag'] = 1
    return x, cnt


# ============================================================
# 10) Setup helpers
# ============================================================

def make_initial_guess(n, device="cpu", mode="data", b=None):
    if mode == "zeros":
        img = torch.zeros((n, n), dtype=torch.float64, device=device)
    elif mode == "data":
        if b is None:
            raise ValueError("b must be provided when mode='data'")
        img = b.detach().clone()
    else:
        raise ValueError("mode must be 'zeros' or 'data'")

    return TorchDictVector(OrderedDict(img=img))


def build_tv_saturation_problem(
    n=128,
    sigma=2.0,
    kernel_size=3,
    noise_level=0.01,
    lam=1e-3,
    device="cpu",
    x0_mode="data"
):
    x_true, b, A_apply, AT_apply, _ = make_shepp_logan_data(
        n=n,
        sigma=sigma,
        kernel_size=kernel_size,
        noise_level=noise_level,
        device=device,
    )

    obj_smooth = SaturationLeastSquares(
        b=b,
        A_apply=A_apply,
        AT_apply=AT_apply,
        x_true=x_true,
        key="img",
        kink_tol=1e-8,
    )

    obj_nonsmooth = TVNonsmooth(
        lam=lam,
        key="img",
        prox_max_iter=2000,
        prox_tol=1e-10,
    )

    problem = Problem(
        obj_smooth=obj_smooth,
        obj_nonsmooth=obj_nonsmooth,
        var={"useEuclidean": True},
    )

    x0 = make_initial_guess(n=n, device=device, mode=x0_mode, b=b)
    return x_true, b, x0, problem

def build_l1_saturation_problem(
    n=128,
    sigma=2.0,
    kernel_size=9,
    noise_level=0.01,
    lam=1e-3,
    device="cpu",
    x0_mode="data"
):
    x_true, b, A_apply, AT_apply, _ = make_shepp_logan_data(
        n=n,
        sigma=sigma,
        kernel_size=kernel_size,
        noise_level=noise_level,
        device=device,
    )

    obj_smooth = SaturationLeastSquares(
        b=b,
        A_apply=A_apply,
        AT_apply=AT_apply,
        x_true=x_true,
        key="img",
        kink_tol=1e-8,
    )

    obj_nonsmooth = L1Nonsmooth(
        lam=lam,
        key="img",
    )

    problem = Problem(
        obj_smooth=obj_smooth,
        obj_nonsmooth=obj_nonsmooth,
        var={"useEuclidean": True},
    )

    x0 = make_initial_guess(n=n, device=device, mode=x0_mode, b=b)
    return x_true, b, x0, problem

def build_wavelet_l1_saturation_problem(
    n=128,
    sigma=2.0,
    kernel_size=3,
    noise_level=0.0,
    lam=2e-4,
    device="cpu",
    x0_mode="zeros",
    wavelet="db2",
    level=2,
    #lam_approx = 1e-4,
):
    x_true, b, A_apply, AT_apply, _ = make_shepp_logan_data(
        n=n,
        sigma=sigma,
        kernel_size=kernel_size,
        noise_level=noise_level,
        device=device,
    )

    obj_smooth = SaturationLeastSquares(
        b=b,
        A_apply=A_apply,
        AT_apply=AT_apply,
        x_true=x_true,
        key="img",
        kink_tol=1e-8,
        mu_smooth = 0.001,
        use_smoothed_kink = True,
        use_full_hessian = False
    )

    obj_nonsmooth = WaveletL1DetailNonsmooth(
        lam=lam,
        key="img",
        wavelet=wavelet,
        level=level,
        mode="periodization",
        #lam_approx = 1e-4,
    )

    problem = Problem(
        obj_smooth=obj_smooth,
        obj_nonsmooth=obj_nonsmooth,
        var={"useEuclidean": True},
    )

    x0 = make_initial_guess(n=n, device=device, mode=x0_mode, b=b)
    return x_true, b, x0, problem

def show_results(x_true, b, x_opt):
    xt = x_true.detach().cpu().numpy()
    bb = b.detach().cpu().numpy()
    rec = x_opt.td["img"].detach().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(xt, cmap="gray")
    plt.title("True image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(bb, cmap="gray")
    plt.title("Observed data")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(rec, cmap="gray")
    plt.title("Reconstruction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

@torch.no_grad()
def check_smooth_derivatives(problem,x, key="img",seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    obj=problem.obj_smooth
    s = x.zero_like()
    s.td[key]=torch.randn_like(x.td[key])
    s_norm = problem.pvector.norm(s)
    if s_norm == 0:
        raise RuntimeError("Random direction has zero norm")
    s = (1.0/s_norm)*s
    
    f0, _ = obj.value(x,1e-12)
    g0, _ = obj.gradient(x,1e-12)
    Hs, _ = obj.hessVec(s,x,1e-12)
    g0s = problem.pvector.dot(g0,s)
    print("n=== First-order derivative check ===")
    print("eps    ||g(x+eps s)-g(x)-eps Hs|| ratio/eps^2")
    for eps in [1e-1, 5e-2,1e-2,5e-3,1e-3,5e-4,1e-4]:
        xeps = x+eps*s
        geps,_ = obj.gradient(xeps,1e-12)
        rem = geps-g0-eps*Hs
        err2 = problem.pvector.norm(rem)
        print(f"{eps:8.1e}   {err2:26.6e} {err2/(eps**2):14.6e}")
        
@torch.no_grad()
def check_formula_interior(problem, x, key="img"):
    obj = problem.obj_smooth
    u = x.td[key]
    Ax = obj.A_apply(u)

    print("\n=== Interior formula check ===")
    print("min(Ax) =", torch.min(Ax).item(), " max(Ax) =", torch.max(Ax).item())

    g, _ = obj.gradient(x, 1e-12)
    g_exact = obj.AT_apply(Ax - obj.b)
    errg = torch.norm(g.td[key] - g_exact).item()
    print("||g - A^T(Ax-b)|| =", errg)

    s = x.zero_like()
    s.td[key] = torch.randn_like(u)
    Hs, _ = obj.hessVec(s, x, 1e-12)
    Hs_exact = obj.AT_apply(obj.A_apply(s.td[key]))
    errH = torch.norm(Hs.td[key] - Hs_exact).item()
    print("||Hs - A^T A s|| =", errH)
    
    
    
        
@torch.no_grad()
def make_interior_test_point(n,device="cpu"):
    img = 0.5+0.05*torch.randn((n,n),dtype=torch.float64,device=device)
    img = torch.clamp(img,0.2,0.8)
    return TorchDictVector(OrderedDict(img=img))

@torch.no_grad()
def check_clipping_margin(problem, x, key="img"):
    obj = problem.obj_smooth
    u = x.td[key]
    Ax = obj.A_apply(u)

    min_to_0 = torch.min(Ax).item()
    min_to_1 = torch.min(1.0 - Ax).item()

    print("\n=== Clipping margin diagnostic ===")
    print("min(Ax)      =", min_to_0)
    print("min(1 - Ax)  =", min_to_1)
    print("If both are comfortably positive, you're away from kinks.")

def plot_tr_history(cnt):
    obj = np.array(cnt.get("objhist", []), dtype=float)
    gnm = np.array(cnt.get("gnormhist", []), dtype=float)
    delt = np.array(cnt.get("deltahist", []), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    if len(obj) > 0:
        axes[0].plot(obj)
        axes[0].set_title("Objective")
        axes[0].set_xlabel("iter")

    if len(gnm) > 0:
        axes[1].plot(gnm)
        axes[1].set_title("prox-gradient norm")
        axes[1].set_xlabel("iter")
        axes[1].set_yscale("log")

    if len(delt) > 0:
        axes[2].plot(delt)
        axes[2].set_title("TR radius")
        axes[2].set_xlabel("iter")
        axes[2].set_yscale("log")

    plt.show()    
# ============================================================
# 11) Main
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cpu"

    # A stable first run:
    n = 64
    sigma = 1.5
    kernel_size = 3
    noise_level = 0.0
    lam = 5e-3

    #x_true, b, x0, problem = build_tv_saturation_problem(
    #    n=n,
    #    sigma=sigma,
    #    kernel_size=kernel_size,
    #    noise_level=noise_level,
    #    lam=lam,
    #    device=device,
    #    x0_mode="data",
    #)
    
    #x_true, b, x0, problem = build_l1_saturation_problem(
    #    n = n,
    #    sigma = sigma,
    #    kernel_size = kernel_size,
    #    noise_level = noise_level,
    #    lam = lam,
    #    device = device,
    #    x0_mode = "data",
    #    )
    x_true, b, x0, problem = build_wavelet_l1_saturation_problem(
        n = n,
        sigma = sigma,
        kernel_size = kernel_size,
        noise_level = noise_level,
        lam = lam,
        device = device,
        x0_mode = "data",
        wavelet = "haar",#"haar",#"db1", #"db2",
        level = 3, 
        #lam_approx = 1e-4,
        )

    params = {
        "spsolver": "SPG2",      # "NCG" or "SPG2"
        "useGCP": True,
        "maxit": 5000,
        "delta": 1.0,
        "gtol": 1e-5,
        "ocScale": 1.0,
        "eta1": 1e-4,
        "eta2": 0.5,
        "gamma1": 0.5,
        "gamma2": 1.5,
        "maxitsp": 20,
        "pred_abs_tol": 1e-12,
        "pred_rel_tol": 1e-12,
        "pred_small_max": 5,
        "outFreq": 1,
        "useInexactGrad": False,
    }
    
    x_test = make_interior_test_point(n=n,device=device)
    check_smooth_derivatives(problem,x_test,key="img",seed=0)
    check_clipping_margin(problem,x_test)
    check_formula_interior(problem,x_test)

    x_opt, cnt, x_best = trustregion(x0, Deltai=1.0, problem=problem, params=params)

    print("\nFinal status flag:", cnt["iflag"])
    print("Iterations:", cnt["iter"])
    print("Final objective:", cnt["objhist"][-1])
    print("Final relative L2 error:", problem.obj_smooth.relative_L2_error(x_opt))

    show_results(x_true, b, x_best)
    plot_tr_history(cnt)
