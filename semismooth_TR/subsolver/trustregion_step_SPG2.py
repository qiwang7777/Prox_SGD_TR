import math
import numpy as np
import copy
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