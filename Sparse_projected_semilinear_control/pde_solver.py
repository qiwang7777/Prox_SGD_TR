import torch

def projection(y,a,b): return torch.clamp(y,min=float(a),max=float(b))

def projection_derivative(y,a,b,zero_selection=0.5,tol=1e-12):
    th = torch.zeros_like(y)
    th[(y>a+tol)&(y<b-tol)] = 1.0
    kink = (torch.abs(y-a)<=tol)|(torch.abs(y-b)<=tol)
    th[kink] = float(zero_selection)
    return th

def solve_state(u,A,a,b,y0=None,tol=1e-10,maxit=50,zero_selection=0.5):
    y = torch.zeros_like(u) if y0 is None else y0.clone()
    for it in range(maxit):
        r = A@y + projection(y,a,b) - u
        rn = float(torch.linalg.vector_norm(r).item())
        if rn <= tol: return y,it,True
        th = projection_derivative(y,a,b,zero_selection)
        M = A + torch.diag(th.reshape(-1))
        s = torch.linalg.solve(M,-r)
        t = 1.0
        for _ in range(20):
            yt = y+t*s
            rt = A@yt + projection(yt,a,b) - u
            if float(torch.linalg.vector_norm(rt).item()) <= (1-1e-4*t)*rn:
                y = yt; break
            t *= 0.5
        else:
            y = y+s
    return y,maxit,False

def solve_adjoint(y,yd,A,a,b,zero_selection=0.5):
    th = projection_derivative(y,a,b,zero_selection)
    return torch.linalg.solve(A+torch.diag(th.reshape(-1)), y-yd)

def solve_linearized_state(v,y,A,a,b,zero_selection=0.5):
    th = projection_derivative(y,a,b,zero_selection)
    return torch.linalg.solve(A+torch.diag(th.reshape(-1)), v)
