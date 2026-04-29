
def directional_fd_value(obj, theta, v, eps):
    th_p = theta.copy()
    th_m = theta.copy()
    th_p.axpy(eps, v)
    th_m.axpy(-eps, v)
    Jp, _ = obj.value(th_p)
    Jm, _ = obj.value(th_m)
    return (Jp - Jm) / (2.0 * eps)


def directional_grad_dot(obj, theta, v):
    g, _ = obj.gradient(theta)
    return g.dot(v)


def grad_check(obj, theta, ntests=3, eps_list=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    for t in range(ntests):
        v = theta.randn_like()
        v.normalize_()
        gTv = directional_grad_dot(obj, theta, v)
        print(f"\nTest {t}: g^T v = {gTv:+.6e}")
        for eps in eps_list:
            fd = directional_fd_value(obj, theta, v, eps)
            relerr = abs(fd - gTv) / max(1.0, abs(fd), abs(gTv))
            print(f" eps={eps: >8.1e} FD={fd:+.6e} relerr={relerr:.3e}")


def directional_fd_grad(obj, theta, v, eps):
    th_p = theta.copy()
    th_m = theta.copy()
    th_p.axpy(eps, v)
    th_m.axpy(-eps, v)
    gp, _ = obj.gradient(th_p)
    gm, _ = obj.gradient(th_m)
    out = gp.copy()
    out.axpy(-1.0, gm)
    out.scal(1.0 / (2.0 * eps))
    return out


def hv_check(obj, theta, ntests=3, eps_list=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    for t in range(ntests):
        v = theta.randn_like()
        v.normalize_()
        Hv, _ = obj.hessVec(v, theta)
        print(f"\nTest {t}: ||Hv|| = {Hv.norm():.6e}")
        for eps in eps_list:
            fdHv = directional_fd_grad(obj, theta, v, eps)
            diff = fdHv.copy()
            diff.axpy(-1.0, Hv)
            relerr = diff.norm() / max(1.0, fdHv.norm(), Hv.norm())
            print(f" eps={eps:>8.1e} ||FD-Hv||/scale={relerr:.3e}")


