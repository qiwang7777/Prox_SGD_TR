import torch.nn as nn
import torch, math
from .Poisson_obj import b_factor


class QuadraticReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5*torch.relu(x)**2
    

class MLP(nn.Module):
    """
    Plain MLP: x -> h(f(x))
    Activation can be 'relu' or 'softplus' etc.
    """
    def __init__(self, in_dim=2, width=64, depth=3, out_dim=1, activation="quadratic"):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "softplus":
            act = nn.Softplus(beta=1.0)   # smooth ReLU-ish
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "quadratic":
            act = QuadraticReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        layers += [nn.Linear(in_dim, width), act]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act]
        layers += [nn.Linear(width, out_dim)]

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # Kaiming init works well for ReLU/Softplus/GELU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in = m.weight.size(1)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        return self.net(x)


class PoissonNet(nn.Module):
    """
    u_theta(x) = b(x) * com_theta(x)
    so Dirichlet BC u=0 is satisfied automatically.
    """
    def __init__(self, in_dim=2, width=64, depth=3,out_dim=1, activation="quadratic"):
        super().__init__()
        self.com = MLP(in_dim=in_dim, width=width, depth=depth, out_dim=1, activation=activation)
        #self.smooth_beta = 50.0

    def forward(self, xy, smooth = False):
        return b_factor(xy) * self.com(xy) 
