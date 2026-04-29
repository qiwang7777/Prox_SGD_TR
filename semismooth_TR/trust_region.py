import torch
import numpy as np
import copy
import time
from .subsolver.trustregion_step_NCG import trustregion_step_NCG
from .subsolver.trustregion_step_SPG2 import trustregion_step_SPG2
from collections import deque


def trustregion(x0, Deltai, problem, params):
    start_time = time.time()

    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 500)
    params.setdefault('gtol', 1e-7)
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

    cnt = {
        'AlgType': f"TR-{params.get('spsolver','NCG')}",
        'iter': 0,
        'nobj1': 0,
        'ngrad': 0,
        'nobj2': 0,
        'nprox': 0,
        'nhess': 0,
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
        'graderr': [],
        'gradtol': []
    }

    obj = problem.obj_smooth

    if params['initProx']:
        x = problem.obj_nonsmooth.prox(x0, 1.0)
        cnt['nprox'] += 1
    else:
        x = x0.copy()

    obj.update(x, "init")

    rej_count = 0
    small_pred_count = 0

    val_true, _ = obj.value(x, 1e-12)
    cnt['nobj1'] += 1

    val_model = val_true
    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)

    phi = problem.obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    Facc = [val_true + phi]
    Fhist = deque(maxlen=params["nonmono_M"])
    Fhist.append(val_true + phi)

    print(f"TR method using {params.get('spsolver','NCG')} Subproblem Solver")
    print("  iter    value    gnorm    del    snorm    nobjs    ngrad    nobjn    nprox     iterSP    flagSP")
    print(f"{0:4d} {0:4d} {val_true+phi:8.6e} {params['delta']:8.6e}  ---      "
          f"{cnt['nobj1']:6d}      {cnt['ngrad']:6d}      {cnt['nobj2']:6d}      {cnt['nprox']:6d} --- ---")

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
    if params['reltol']:
        gtol = params['gtol'] * gnorm

    for i in range(1, params['maxit'] + 1):
        params['tolsp'] = min(params['atol'], params['rtol'] * (gnorm ** params['spexp']))
        grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)

        val_model = val_true
        cnt['nobj1'] += 1

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
            Facc.append(val_true + phi)
            Fhist.append(val_true + phi)

            relL2u = obj.relative_L2_error_control(x)
            relL2y = obj.relative_L2_error_state(x)
            print(f"   relL2(control)={relL2u:.3e}, relL2(state)={relL2y:.3e}")

            if rho > params['eta2']:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])

        if i % params['outFreq'] == 0:
            print(f"{i:4d}    {val_true + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      "
                  f"{cnt['nobj1']:6d}     {cnt['ngrad']:6d}       {cnt['nobj2']:6d}     {cnt['nprox']:6d}      "
                  f"{iter_count:4d}        {iflag:1d}")

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
        stol_abs   = params["stol_abs"]
        K          = params["stag_window"]
        ftol_rel   = params["ftol_rel"]
        max_reject = params["max_reject"]

        stop_grad = (gnorm <= gtol)
        stop_step = (snorm < stol_abs) and (params["delta"] <= delta_stop)
        stop_stag = False
        if len(Facc) >= K + 1:
            Fold = Facc[-(K+1)]
            Fnew = Facc[-1]
            rel_change = abs(Fold - Fnew) / max(1.0, abs(Fnew))
            stop_stag = (rel_change < ftol_rel)
        stop_stuck = (params["delta"] <= 10 * delta_stop and rej_count >= max_reject)
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
                reason = "trust region stuck"
            else:
                flag = 1
                reason = "maximum iterations reached"

            cnt['iter'] = i
            cnt['timetotal'] = time.time() - start_time
            cnt['iflag'] = flag

            print("Optimization terminated because", reason)
            print(f"Total time: {cnt['timetotal']:8.6e} seconds")
            return x, cnt

    cnt['iter'] = params['maxit']
    cnt['timetotal'] = time.time() - start_time
    cnt['iflag'] = 1
    return x, cnt

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
