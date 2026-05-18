def compute_gradient(x, problem, params, cnt):
    gtol = 1e-12
    grad, gerr = problem.obj_smooth.gradient(x, gtol)
    cnt['ngrad'] += 1
    dgrad = problem.dvector.dual(grad)
    pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
    cnt['nprox'] += 1
    gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']

    params['gradTol'] = gtol
    cnt.setdefault('graderr', []).append(gerr)
    cnt.setdefault('gradtol', []).append(gtol)
    return grad, dgrad, gnorm, cnt
