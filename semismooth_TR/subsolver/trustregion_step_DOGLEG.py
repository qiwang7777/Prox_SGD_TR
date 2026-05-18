import numpy as np
import torch


def _vec_from_data(template, data):
    out = template.zero_like()
    out.data = data.clone().reshape_as(out.data)
    return out


def _vec_from_flat(template, flat_data):
    out = template.zero_like()
    out.data = flat_data.clone().reshape(-1, 1)
    return out


def dogleg_active_free_sets(x, dgrad, problem, tol=1e-12):
    u_a, u_b = problem.obj_nonsmooth.get_parameter()

    xdat = x.data.reshape(-1, 1)
    gdat = dgrad.data.reshape(-1, 1)

    active_lower = (xdat <= u_a + tol) & (gdat > 0.0)
    active_upper = (xdat >= u_b - tol) & (gdat < 0.0)

    active = active_lower | active_upper
    free = ~active

    return active.reshape(-1), free.reshape(-1)


def hessvec_free(vF, x, free_idx, problem, params, cnt):
    vfull = x.zero_like()
    vfull.data.reshape(-1, 1)[free_idx] = vF.data.reshape(-1, 1)

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
    tol=1e-8,
    maxit=50,
    reg=1e-8,
    debug=False,
):
    x = b.zero_like()

    def Areg(v):
        Av = apply_A(v)
        out = Av.copy()
        if reg != 0.0:
            out.axpy(reg, v)
        return out

    r = b - Areg(x)
    p = r.copy()
    rsold = r.dot(r)

    if rsold <= tol * tol:
        return x, 0, np.sqrt(max(rsold, 0.0))

    k = 0

    for k in range(1, maxit + 1):
        Ap = Areg(p)
        pAp = p.dot(Ap)

        if debug:
            print("DOGLEG-CG iter", k, "pAp =", pAp, "||r|| =", np.sqrt(max(rsold, 0.0)))

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


def embed_free_step(sF, free_idx, x):
    sfull = x.zero_like()
    sfull.data.reshape(-1, 1)[free_idx] = sF.data.reshape(-1, 1)
    return sfull


def quadratic_model_reduction(x, s, dgrad, problem, params, cnt):
    Hs, _ = problem.obj_smooth.hessVec(s, x, params["gradTol"])
    cnt["nhess"] = cnt.get("nhess", 0) + 1

    gs = problem.pvector.dot(dgrad, s)
    sHs = problem.dvector.apply(Hs, s)

    pRed = -(gs + 0.5 * sHs)
    return float(pRed), cnt


def project_box(x, problem):
    u_a, u_b = problem.obj_nonsmooth.get_parameter()
    return _vec_from_data(
        x,
        torch.clamp(x.data, min=u_a, max=u_b),
    )


def dogleg_step_free(x, dgrad, free_idx, problem, params, cnt):
    delta = params["delta"]
    nf = len(free_idx)

    if nf == 0:
        return x.zero_like(), 0.0, cnt, "all_active", 0

    gF_data = dgrad.data.reshape(-1, 1)[free_idx].clone()
    gF = _vec_from_flat(x, gF_data)

    gnorm = problem.pvector.norm(gF)

    if gnorm <= 1e-30:
        return x.zero_like(), 0.0, cnt, "zero_gradient", 0

    def apply_HF(vF):
        nonlocal cnt
        HvF, cnt = hessvec_free(vF, x, free_idx, problem, params, cnt)
        return HvF

    HgF = apply_HF(gF)
    gHg = problem.pvector.dot(gF, HgF)

    if gHg > params["dogleg_safeguard"] * gnorm * gnorm:
        alpha_c = (gnorm * gnorm) / gHg
        sC = (-alpha_c) * gF
    else:
        sC = (-(delta / max(gnorm, 1e-30))) * gF

    nC = problem.pvector.norm(sC)

    if nC >= delta:
        sC.scal(delta / max(nC, 1e-30))
        s = embed_free_step(sC, free_idx, x)
        return s, problem.pvector.norm(s), cnt, "cauchy_boundary", 0

    rhs = (-1.0) * gF

    sN, cg_it, cg_res = cg_solve_free(
        apply_HF,
        rhs,
        tol=params["dogleg_cg_tol"],
        maxit=params["dogleg_cg_maxit"],
        reg=params["dogleg_reg"],
        debug=params.get("debug", False),
    )

    nN = problem.pvector.norm(sN)

    if nN <= delta:
        s = embed_free_step(sN, free_idx, x)
        return s, problem.pvector.norm(s), cnt, "newton", cg_it

    p = sN - sC
    a = problem.pvector.dot(p, p)
    b = 2.0 * problem.pvector.dot(sC, p)
    c = problem.pvector.dot(sC, sC) - delta * delta

    disc = max(b * b - 4.0 * a * c, 0.0)

    tau = (-b + np.sqrt(disc)) / max(2.0 * a, 1e-30)
    tau = float(np.clip(tau, 0.0, 1.0))

    sDL = sC + tau * p

    s = embed_free_step(sDL, free_idx, x)
    return s, problem.pvector.norm(s), cnt, "dogleg_boundary", cg_it


def trustregion_step_DOGLEG(x, val, dgrad, phi, problem, params, cnt):
    """
    Matrix-free Dogleg subproblem solver for box-constrained controls.

    Same interface as NCG/SPG2/SSN:

        s, snorm, pRed, phinew, iflag, iter_count, cnt, params

    Flags:
        0: zero free gradient
        1: interior Newton step
        2: boundary Cauchy/Dogleg step
        3: all variables active
    """
    params.setdefault("debug", False)
    params.setdefault("gradTol", np.sqrt(np.finfo(float).eps))
    params.setdefault("delta", 1.0)

    params.setdefault("dogleg_reg", 1e-8)
    params.setdefault("dogleg_cg_tol", 1e-8)
    params.setdefault("dogleg_cg_maxit", 50)
    params.setdefault("dogleg_safeguard", 1e-12)

    active, free = dogleg_active_free_sets(
        x,
        dgrad,
        problem,
        tol=params.get("dogleg_active_tol", 1e-12),
    )

    free_idx = torch.where(free)[0]

    s, snorm, cnt, step_type, cg_it = dogleg_step_free(
        x,
        dgrad,
        free_idx,
        problem,
        params,
        cnt,
    )

    xtrial = x + s
    xproj = project_box(xtrial, problem)
    s = xproj - x
    snorm = problem.pvector.norm(s)

    phinew = problem.obj_nonsmooth.value(xproj)
    cnt["nobj2"] = cnt.get("nobj2", 0) + 1

    pRed, cnt = quadratic_model_reduction(
        x,
        s,
        dgrad,
        problem,
        params,
        cnt,
    )

    if step_type == "zero_gradient":
        iflag = 0
    elif step_type == "newton":
        iflag = 1
    elif step_type == "all_active":
        iflag = 3
    else:
        iflag = 2

    if params.get("debug", False):
        print("---- DOGLEG DEBUG ----")
        print("free size =", int(free.sum().item()))
        print("active size =", int(active.sum().item()))
        print("step_type =", step_type)
        print("snorm =", snorm)
        print("pRed =", pRed)
        print("cg_it =", cg_it)
        print("----------------------")

    iter_count = cg_it if cg_it > 0 else 1

    cnt["dogleg_step_type_last"] = step_type
    cnt["dogleg_cg_last"] = cg_it

    return s, snorm, pRed, phinew, iflag, iter_count, cnt, params
