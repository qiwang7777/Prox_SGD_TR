import math
import numpy as np
import copy
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
            pR = p0
            nR = n0
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
                    pp = p0
                    t0 = t1
                    n0 = n1
                    p0 = t0 * (0.5 * t0 * kappa + gs) + n0 - nold
                    if p0 >= pp:
                        tL = tp
                        pL = pp
                        break
            tR = t0
            pR = p0
            nR = n0
    else:
        pR = tR * (0.5 * tR * kappa + gs) + nR - nold

    t = tR
    if kappa > 0:
        t = min(tR, -(((nR - nold) / max(tR - tL, 1e-30)) + gs) / kappa)

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
        return t, nold, cnt

    if pt >= max(pL, pR):
        return tR, nR, cnt

    v = tL
    pv = pL
    w = tR
    pw = pR
    d = 0.0
    e = max(t - tL, tR - t)
    tm = 0.5 * (tL + tR)

    for _ in range(maxit):
        dL = tL - t
        dR = tR - t
        tol1 = tol0 * abs(t) + eps1
        tol2 = 2 * tol1
        tol3 = eps0 * max(abs(Qt), 1.0)

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
                e = dR if t <= tm else dL
                d = lam * e
            else:
                d = p / q
                u = t + d
                if (u - tL < tol2) or (tR - u < tol2):
                    d = tol1 if tm >= t else -tol1
        else:
            e = dR if t <= tm else dL
            d = lam * e

        u = t + d
        if abs(d) < tol1:
            u = t + tol1 if d >= 0 else t - tol1

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
            v, pv = w, pw
            w, pw = t, pt
            t, pt = u, pu
            nt, Qt = nu, Qu
        else:
            if u < t:
                tL = u
            else:
                tR = u
            if pu <= pw or w == t:
                v, pv = w, pw
                w, pw = u, pu
            elif pu <= pv or v == t or v == w:
                v, pv = u, pu

        tm = 0.5 * (tL + tR)

        if pt <= (mu * min(0.0, Qt) + tol3) and abs(t - tm) <= (tol2 - 0.5 * (tR - tL)):
            break

    return t, nt, cnt


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

    Hg, _ = problem.obj_smooth.hessVec(dgrad, x, params["gradTol"])
    cnt["nhess"] = cnt.get("nhess", 0) + 1

    gHg = problem.dvector.apply(Hg, dgrad)
    gg = problem.pvector.dot(dgrad, dgrad)

    if gHg > params["safeguard"] * gg:
        lambdaTmp = gg / gHg
    else:
        lambdaTmp = params["t"] / np.sqrt(max(gg, 1e-30))

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
    phiold = phi

    for iter0 in range(1, params["maxitsp"] + 1):
        Hs, _ = problem.obj_smooth.hessVec(s, x, params["gradTol"])
        cnt["nhess"] = cnt.get("nhess", 0) + 1
        sHs = problem.dvector.apply(Hs, s)

        ds = problem.pvector.dot(s, y - x)
        ss = snorm ** 2
        alphaMax = (-ds + np.sqrt(max(ds**2 + ss * (del2 - ss0), 0.0))) / max(ss, 1e-30)

        alpha, phiold, cnt = dbls(
            phiold, y, s, alphaMax, lam1, sHs, gs,
            params["maxitdbls"], problem, cnt
        )

        y = y + alpha * s
        gmod = gmod + alpha * problem.dvector.dual(Hs)
        valold = valold + alpha * (gs + 0.5 * alpha * sHs)

        ss0 = alpha**2 * ss + 2 * alpha * ds + ss0
        snorm0 = np.sqrt(max(ss0, 0.0))

        if snorm0 >= (1 - params["safeguard"]) * params["delta"]:
            iflag = 2
            iter_count = iter0
            break

        if sHs > params["safeguard"] * ss:
            lambdaTmp = ss / sHs
        else:
            lambdaTmp = params["t"] / max(problem.pvector.norm(gmod), 1e-30)

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
        d = dx0 - dx

        if params["ncg_type"] == 4:
            denom = problem.pvector.dot(s, d)
            beta = max(0.0, gnorm2 / max(denom, 1e-30))
        else:
            beta = 0.0

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
            s = copy.deepcopy(dx)
            lam1 = lambda_
            gs = problem.pvector.dot(gmod, s)

        snorm = problem.pvector.norm(s)
        iter_count = iter0

    s = y - x
    snorm = problem.pvector.norm(s)
    phinew = phiold
    pRed = (val + phi) - (valold + phinew)

    return s, snorm, pRed, phinew, iflag, iter_count, cnt, params