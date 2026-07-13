import torch
import numpy as np
import copy
import time
from collections import deque

from .subsolver.trustregion_step_NCG import trustregion_step_NCG
from .subsolver.trustregion_step_SPG2 import trustregion_step_SPG2
from .subsolver.trustregion_step_SSN import trustregion_step_SSN
from .subsolver.trustregion_step_DOGLEG import trustregion_step_DOGLEG

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
    params.setdefault('deltamin', 1e-8)
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

    # Smoothing controls
    params.setdefault("mu_smooth", 1e-4)
    params.setdefault("mu_min",1e-14)
    params.setdefault("mu_power",1.0)
    params.setdefault("mu_factor",1.0)
    params.setdefault("boundary_tol", 0.8)
    params.setdefault("smooth_mode",False)
    # Trigger smoothing when ||d_k|| / v_k is too small.
    params.setdefault("small_step_ratio_tol", 1e-1)
    
    params.setdefault("delta_floor", 1e-14)

    cnt = {
        'AlgType': f"TR-{params.get('spsolver', 'SPG2')}",
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
        'gradtol': [],
        'gradtypehist': [],
        'muhist': [],
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

    print(f"TR method using {params.get('spsolver', 'SPG2')} Subproblem Solver")
    print(
        "  iter    value         gnorm        del          snorm        "
        "nobjs    ngrad    nobjn    nprox     iterSP    flagSP    gradtype"
    )
    print(
        f"{0:4d}    {val_true + phi:12.6e}    {gnorm:8.6e}    "
        f"{params['delta']:8.6e}    ---      "
        f"{cnt['nobj1']:6d}    {cnt['ngrad']:6d}    "
        f"{cnt['nobj2']:6d}    {cnt['nprox']:6d}    "
        f"---    ---    {params.get('grad_type', '---')}"
    )

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
        params['tolsp'] = min(
            params['atol'],
            params['rtol'] * (gnorm ** params['spexp'])
        )

        grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)

        if hasattr(obj, "value_model"):
            val_model, _ = obj.value_model(x, 1e-12)
        else:
            val_model = val_true
        cnt['nobj1'] += 1

        solver = params.get('spsolver', 'SPG2').upper()
        delta_old = params['delta']

        def solve_subproblem(grad_, dgrad_, val_model_):
            if solver == 'SPG2':
                return trustregion_step_SPG2(
                    x, val_model_, grad_, dgrad_, phi, problem, params, cnt
                )
            if solver == 'NCG':
                return trustregion_step_NCG(
                    x, val_model_, dgrad_, phi, problem, params, cnt
                )
            if solver == 'SSN':
                return trustregion_step_SSN(
                    x, val_model_, dgrad_, phi, problem, params, cnt
                )
            return trustregion_step_DOGLEG(
                x, val_model_, dgrad_, phi, problem, params, cnt
            )

        # First solve using the currently active model.
        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = solve_subproblem(
            grad, dgrad, val_model
        )

        step_ratio = snorm / max(float(gnorm), 1e-300)
        smoothing_triggered = (
            params.get("grad_type") != "smooth"
            and gnorm > gtol
            and step_ratio <= params["small_step_ratio_tol"]
        )

        if smoothing_triggered:
            # The nonsmooth candidate step is too small relative to v_k.
            # Switch to the smoothed model and resolve the same TR subproblem.
            params["smooth_mode"] = True
            grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)

            if hasattr(obj, "value_model"):
                val_model, _ = obj.value_model(x, 1e-12)
                cnt["nobj1"] += 1
            else:
                val_model = val_true

            s, snorm, pRed, phinew, iflag, iter_count, cnt, params = solve_subproblem(
                grad, dgrad, val_model
            )
            step_ratio = snorm / max(float(gnorm), 1e-300)

        pRed = float(pRed)

        pred_floor = max(
            params["pred_abs_tol"],
            params["pred_rel_tol"] * max(1.0, abs(val_model + phi))
        )

        if pRed <= pred_floor:
            small_pred_count += 1
        else:
            small_pred_count = 0

        if (
            small_pred_count >= params["pred_small_max"]
            and params["delta"] <= 10.0 * params["delta_stop"]
        ):
            cnt['iter'] = i
            cnt['timetotal'] = time.time() - start_time
            cnt['iflag'] = 6
            print("Optimization terminated because predicted reduction is tiny repeatedly.")
            print(f"Total time: {cnt['timetotal']:8.6e} seconds")
            return x, cnt, best_x

        xnew = x + s

        valnew_true, _ = obj.value(xnew, 1e-12)
        cnt['nobj1'] += 1

        phinew_true = problem.obj_nonsmooth.value(xnew)
        cnt['nobj2'] += 1

        aRed = (val_true + phi) - (valnew_true + phinew_true)
        rho = -np.inf if pRed <= 0.0 else float(aRed) / pRed

        Fref = max(Fhist)
        accept_nm = (valnew_true + phinew_true) <= (Fref - 1e-12)
        accept = (rho > params['eta1']) and accept_nm

        print(
            "debug:",
            "aRed=", float(aRed),
            "pRed=", float(pRed),
            "rho=", float(rho),
            "gradtype=", params.get("grad_type", "---"),
            "mu=", params.get("active_mu_smooth", None),
        )
        
        boundary_ratio = snorm/max(delta_old,1e-300)
        boundary_active = boundary_ratio >= params.get("boundary_tol",0.8)

        if not accept:
            params['delta'] = max(
                params['delta_floor'],
                params['gamma1'] * delta_old
            )

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

            relL2 = obj.relative_L2_error(x)
            print("relative L2 error =", relL2)

            if relL2 < best_relL2:
                best_relL2 = relL2
                best_x = x.copy()
                cnt["best_relL2"] = best_relL2
                cnt["best_iter"] = i

            if rho <= params['eta2']:
                params['delta'] = delta_old
            else:
                if boundary_active:
                    params["delta"] = min(params["deltamax"], max(params["deltamin"],params["gamma2"]*delta_old))
                else:
                    params["delta"] = max(params["deltamin"], delta_old)

            # Smoothing is used only as a local recovery mechanism.
            # After a successful smoothed step, return to the nonsmooth model.
            if params.get("grad_type") == "smooth":
                params["smooth_mode"] = False
                                          

        if i % params['outFreq'] == 0:
            print(
                f"{i:4d}    {val_true + phi:12.6e}    {gnorm:8.6e}    "
                f"{params['delta']:8.6e}    {snorm:8.6e}    "
                f"{cnt['nobj1']:6d}    {cnt['ngrad']:6d}    "
                f"{cnt['nobj2']:6d}    {cnt['nprox']:6d}    "
                f"{iter_count:4d}    {iflag:1d}    "
                f"{params.get('grad_type', '---')}"
            )

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

        stop_grad = gnorm <= gtol
        stop_step = (snorm < stol_abs) and (params["delta"] <= delta_stop)
        stop_stuck = params["delta"] <= 10 * delta_stop and rej_count >= max_reject

        stop_stag = False
        if len(Facc) >= K + 1:
            Fold = Facc[-(K + 1)]
            Fnew = Facc[-1]
            rel_change = abs(Fold - Fnew) / max(1.0, abs(Fnew))
            stop_stag = rel_change < ftol_rel

        stop_maxit = i >= params["maxit"]

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
                reason = "trust region stuck: delta small plus repeated rejections"
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

    return x, cnt, best_x


def compute_gradient(x, problem, params, cnt):
    gtol = 1e-12

    # smooth_mode is activated in trustregion() by the step-based test
    #     ||d_k|| <= kappa_d v_k.
    use_smoothing = params.get("smooth_mode", False)

    if use_smoothing:
        mu = max(params["mu_min"],min(params['mu_smooth'],params['mu_factor']*(params['delta']**params['mu_power']),),)
        grad, gerr = problem.obj_smooth.gradient_smooth(x, mu, gtol)
        grad_type = "smooth"
    else:
        grad, gerr = problem.obj_smooth.gradient(x, gtol)
        grad_type = "clarke"
        mu = None

    cnt["ngrad"] += 1

    dgrad = problem.dvector.dual(grad)

    pgrad = problem.obj_nonsmooth.prox(
        x - params["ocScale"] * dgrad,
        params["ocScale"]
    )
    cnt["nprox"] += 1

    gnorm = problem.pvector.norm(pgrad - x) / params["ocScale"]

    params["gradTol"] = gtol
    params["grad_type"] = grad_type
    params["active_mu_smooth"] = mu

    cnt.setdefault("graderr", []).append(gerr)
    cnt.setdefault("gradtol", []).append(gtol)
    cnt.setdefault("gradtypehist", []).append(grad_type)
    cnt.setdefault("muhist", []).append(mu)

    return grad, dgrad, gnorm, cnt
