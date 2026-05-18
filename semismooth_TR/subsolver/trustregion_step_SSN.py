

import numpy as np
import torch
from semismooth_TR.compute_gradient import compute_gradient

def _vec_from_flat(template, flat_data):
    out = template.zero_like()
    out.data = flat_data.clone().reshape_as(out.data)
    return out


def _vec_from_data(template, data):
    out = template.zero_like()
    out.data = data.clone().reshape_as(out.data)
    return out


def projected_residual(x, dgrad, problem, params, cnt):
    px = problem.obj_nonsmooth.prox(x - dgrad, 1.0)
    cnt["nprox"] = cnt.get("nprox", 0) + 1
    return x - px, cnt


def ssn_active_sets(x, dgrad, problem, tol=1e-12):
    """
    Box-constrained active/free sets.

    This part still assumes obj_nonsmooth has get_parameter()
    returning lower/upper bounds.
    """
    u_a, u_b = problem.obj_nonsmooth.get_parameter()

    xdat = x.data.reshape(-1, 1)
    gdat = dgrad.data.reshape(-1, 1)

    z = xdat - gdat

    active_lower = z <= u_a + tol
    active_upper = z >= u_b - tol
    free = ~(active_lower | active_upper)

    return active_lower.reshape(-1), active_upper.reshape(-1), free.reshape(-1)


def hessvec_free(vF, x, free_idx, problem, params, cnt):
    """
    Matrix-free reduced Hessian action:

        vF -> H_FF vF

    No explicit ControlVector construction.
    """
    vfull = x.zero_like()
    vfull_flat = vfull.data.reshape(-1, 1)
    vfull_flat[free_idx] = vF.data.reshape(-1, 1)

    Hv_full, _ = problem.obj_smooth.hessVec(
        vfull,
        x,
        params["gradTol"],
    )
    cnt["nhess"] = cnt.get("nhess", 0) + 1

    HvF_data = Hv_full.data.reshape(-1, 1)[free_idx].clone()
    HvF = _vec_from_flat(vF, HvF_data)

    return HvF, cnt


def cg_solve_free(
    apply_A,
    b,
    x0=None,
    tol=1e-8,
    maxit=50,
    reg=1e-8,
    debug=False,
):
    if x0 is None:
        x = b.zero_like()
    else:
        x = x0.copy()

    def Areg(v):
        Av = apply_A(v)
        out = Av.copy()
        if reg != 0.0:
            out.axpy(reg, v)
        return out

    r = b - Areg(x)
    p = r.copy()
    rsold = r.dot(r)

    if debug:
        print("CG DEBUG")
        print("  ||b||  =", b.norm())
        print("  ||r0|| =", np.sqrt(max(rsold, 0.0)))
        print("  reg    =", reg)

    if rsold <= tol * tol:
        return x, 0, np.sqrt(max(rsold, 0.0))

    k = 0

    for k in range(1, maxit + 1):
        Ap = Areg(p)
        pAp = p.dot(Ap)

        if debug:
            print("  iter =", k)
            print("  ||p|| =", p.norm())
            print("  ||Ap|| =", Ap.norm())
            print("  pAp =", pAp)

        if abs(pAp) <= 1e-30:
            break

        alpha = rsold / pAp

        x.axpy(alpha, p)
        r.axpy(-alpha, Ap)

        rsnew = r.dot(r)

        if rsnew <= tol * tol:
            return x, k, np.sqrt(max(rsnew, 0.0))

        beta = rsnew / rsold
        p.scal(beta)
        p.axpy(1.0, r)

        rsold = rsnew

    return x, k, np.sqrt(max(rsold, 0.0))


def solve_ssn_newton_system_mf(
    x,
    dgrad,
    free_mask,
    active_lower,
    active_upper,
    problem,
    params,
    cnt,
):
    u_a, u_b = problem.obj_nonsmooth.get_parameter()

    xflat = x.data.reshape(-1, 1)
    gflat = dgrad.data.reshape(-1, 1)

    sflat = torch.zeros_like(xflat)

    active_lower = active_lower.reshape(-1)
    active_upper = active_upper.reshape(-1)

    sflat[active_lower, 0] = u_a - xflat[active_lower, 0]
    sflat[active_upper, 0] = u_b - xflat[active_upper, 0]

    free_idx = torch.where(free_mask)[0]

    if len(free_idx) == 0:
        return _vec_from_data(x, sflat.reshape_as(x.data)), 0, 0.0, cnt

    gF_data = gflat[free_idx].clone()
    gF = _vec_from_flat(x, gF_data)
    rhs = (-1.0) * gF

    def apply_A(vF):
        nonlocal cnt
        AvF, cnt = hessvec_free(vF, x, free_idx, problem, params, cnt)
        return AvF

    sF, cg_it, cg_res = cg_solve_free(
        apply_A,
        rhs,
        x0=None,
        tol=params["ssn_cg_tol"],
        maxit=params["ssn_cg_maxit"],
        reg=params["ssn_reg"],
        debug=params.get("debug", False),
    )

    sflat[free_idx, 0] = sF.data.reshape(-1)

    return _vec_from_data(x, sflat.reshape_as(x.data)), cg_it, cg_res, cnt


def ssn_linesearch(
    x,
    s,
    val,
    phi,
    dgrad,
    problem,
    params,
    cnt,
):
    alpha = 1.0
    f0 = val + phi

    c1 = params["ssn_c1"]
    bt = params["ssn_bt"]
    max_ls = params["ssn_bt_maxit"]

    gTs = problem.pvector.dot(dgrad, s)

    for _ in range(max_ls):
        xtrial = x + alpha * s

        # keep same nonsmooth-object style as NCG
        xtrial = problem.obj_nonsmooth.prox(xtrial, 1.0)
        cnt["nprox"] = cnt.get("nprox", 0) + 1

        val_trial, _ = problem.obj_smooth.value(xtrial, 1e-12)
        cnt["nobj1"] = cnt.get("nobj1", 0) + 1

        phi_trial = problem.obj_nonsmooth.value(xtrial)
        cnt["nobj2"] = cnt.get("nobj2", 0) + 1

        if val_trial + phi_trial <= f0 + c1 * alpha * gTs:
            return xtrial - x, val_trial, phi_trial, alpha, cnt

        alpha *= bt

    xtrial = x + alpha * s
    xtrial = problem.obj_nonsmooth.prox(xtrial, 1.0)
    cnt["nprox"] = cnt.get("nprox", 0) + 1

    val_trial, _ = problem.obj_smooth.value(xtrial, 1e-12)
    cnt["nobj1"] = cnt.get("nobj1", 0) + 1

    phi_trial = problem.obj_nonsmooth.value(xtrial)
    cnt["nobj2"] = cnt.get("nobj2", 0) + 1

    if params.get("debug", False):
        print("SSN line search fallback")
        print("  gTs =", gTs)
        print("  alpha =", alpha)
        print("  trial step norm =", (xtrial - x).norm())

    return xtrial - x, val_trial, phi_trial, alpha, cnt


def quadratic_model_reduction_ssn(x, s, dgrad, phi, phinew, problem, params, cnt):
    Hs, _ = problem.obj_smooth.hessVec(s, x, params["gradTol"])
    cnt["nhess"] = cnt.get("nhess", 0) + 1

    gs = problem.pvector.dot(dgrad, s)
    sHs = problem.dvector.apply(Hs, s)

    pRed = phi - phinew - gs - 0.5 * sHs

    return float(pRed), cnt


def trustregion_step_SSN(x, val, dgrad, phi, problem, params, cnt):
    """
    Matrix-free SSN subproblem solver.

    Same interface as trustregion_step_NCG:

        s, snorm, pRed, phinew, iflag, iter_count, cnt, params

    Flags:
        0: projected residual tolerance reached
        1: max SSN iterations reached
        2: trust-region boundary reached
    """
    params.setdefault("debug", False)
    params.setdefault("gradTol", np.sqrt(np.finfo(float).eps))

    params.setdefault("delta", 1.0)

    params.setdefault("ssn_maxit", 1)
    params.setdefault("ssn_tol", 1e-10)
    params.setdefault("ssn_reg", 1e-6)
    params.setdefault("ssn_cg_tol", 1e-6)
    params.setdefault("ssn_cg_maxit", 40)

    params.setdefault("ssn_bt_maxit", 10)
    params.setdefault("ssn_c1", 1e-4)
    params.setdefault("ssn_bt", 0.5)

    x_base = x.copy()
    dgrad_base = dgrad.copy()
    phi_base = phi

    xk = x.copy()
    valk = val
    phik = phi

    Fk, cnt = projected_residual(xk, dgrad, problem, params, cnt)
    Fnrm0 = problem.pvector.norm(Fk)

    if Fnrm0 <= params["ssn_tol"]:
        s = x.zero_like()
        return s, 0.0, 0.0, phi, 0, 0, cnt, params

    total_step = x.zero_like()

    iflag = 1
    iter_count = 0
    cg_total = 0

    for it in range(1, params["ssn_maxit"] + 1):
        active_lower, active_upper, free = ssn_active_sets(
            xk,
            dgrad,
            problem,
        )

        sN, cg_it, cg_res, cnt = solve_ssn_newton_system_mf(
            xk,
            dgrad,
            free,
            active_lower,
            active_upper,
            problem,
            params,
            cnt,
        )

        cg_total += cg_it

        if params.get("debug", False):
            print("---- SSN DEBUG ----")
            print("||F|| =", Fnrm0)
            print("||dgrad|| =", problem.pvector.norm(dgrad))
            print("free size =", int(free.sum().item()))
            print("active lower =", int(active_lower.sum().item()))
            print("active upper =", int(active_upper.sum().item()))
            print("||sN|| before clip =", problem.pvector.norm(sN))
            print("cg_it =", cg_it, "cg_res =", cg_res)
            print("-------------------")

        sn = problem.pvector.norm(sN)

        if sn > params["delta"]:
            sN.scal(params["delta"] / max(sn, 1e-30))

        s_acc, val_new, phi_new, alpha_ls, cnt = ssn_linesearch(
            xk,
            sN,
            valk,
            phik,
            dgrad,
            problem,
            params,
            cnt,
        )

        xk = xk + s_acc
        total_step = xk - x_base

        valk = val_new
        phik = phi_new

        grad_new, dgrad_new, gnorm_new, cnt = compute_gradient(
            xk,
            problem,
            params,
            cnt,
        )

        dgrad = dgrad_new

        Fnew, cnt = projected_residual(xk, dgrad, problem, params, cnt)
        Fnrm = problem.pvector.norm(Fnew)

        iter_count = it

        if params.get("debug", False):
            print("SSN accepted step")
            print("  alpha_ls =", alpha_ls)
            print("  ||total_step|| =", problem.pvector.norm(total_step))
            print("  ||Fnew|| =", Fnrm)

        if Fnrm <= params["ssn_tol"]:
            iflag = 0
            break

        if problem.pvector.norm(total_step) >= (1.0 - 1e-12) * params["delta"]:
            iflag = 2
            break

    s = total_step
    snorm = problem.pvector.norm(s)
    phinew = phik

    pRed, cnt = quadratic_model_reduction_ssn(
        x_base,
        s,
        dgrad_base,
        phi_base,
        phinew,
        problem,
        params,
        cnt,
    )

    cnt["ssn_cg_last"] = cg_total
    cnt["ssn_iflag_last"] = iflag
    cnt["ssn_iter_last"] = iter_count

    return s, snorm, pRed, phinew, iflag, iter_count, cnt, params
