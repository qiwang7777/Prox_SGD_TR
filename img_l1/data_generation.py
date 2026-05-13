from collections import OrderedDict, deque
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import torch
import torch.nn.functional as F

def make_shepp_logan_data(n=128, sigma=2.0, kernel_size=9, noise_level=0.01, device="cpu"):
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