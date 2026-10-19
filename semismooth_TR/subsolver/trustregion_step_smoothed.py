def prox_in_tr_ball(z, t, center, radius, problem, cnt):
    """prox of t*phi plus the indicator of B_radius(center)."""

    def candidate(lam):
        a = 1.0 / (1.0 + lam)
        argument = a * z + (1.0 - a) * center
        y = problem.obj_nonsmooth.prox(argument, a * t)
        cnt["nprox"] += 1
        return y

    def distance(y):
        return problem.pvector.norm(y - center)

    y = candidate(0.0)
    if distance(y) <= radius:
        return y

    lo, hi = 0.0, 1.0
    for _ in range(100):
        y = candidate(hi)
        if distance(y) <= radius:
            break
        hi *= 2.0
    else:
        raise RuntimeError("Cannot bracket a feasible proximal step.")

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        trial = candidate(mid)

        if distance(trial) > radius:
            lo = mid
        else:
            hi, y = mid, trial

    return y

import math


def trustregion_step_smoothed(x, phi, problem, params, cnt):
    obj = problem.obj_smooth
    mu = params["active_mu_smooth"]
    radius = params["delta"]
    maxit = params.get("smooth_maxit", params["maxitsp"])
    tol = max(1e-12, params.get("smooth_tol", params["tolsp"]))

    # x, mu, and B_k remain fixed throughout this subproblem solve.
    fmu0, _ = obj.value_smooth(x, mu, 1e-12)
    cnt["nobj1"] += 1

    def B(step):
        hv, _ = obj.hessVec(step, x, 1e-12)
        cnt["nhess"] = cnt.get("nhess", 0) + 1
        return problem.dvector.dual(hv)

    def increment(step):
        """m_sm(step) - m_sm(0)."""
        y = x + step
        fmu, _ = obj.value_smooth(y, mu, 1e-12)
        phi_y = problem.obj_nonsmooth.value(y)
        cnt["nobj1"] += 1
        cnt["nobj2"] += 1

        quadratic = 0.5 * problem.pvector.dot(step, B(step))
        return float(fmu - fmu0 + quadratic + phi_y - phi)

    step = x.zero_like()
    value = 0.0
    best_step = step.copy()
    best_value = 0.0
    t = params.get("smooth_step_initial", 1.0)
    if t <= 0:
        raise ValueError("smooth_step_initial must be positive.")

    iflag = 1       # Inner iteration limit.
    iter_count = 0

    for iter_count in range(1, maxit + 1):
        y = x + step

        gmu, _ = obj.gradient_smooth(y, mu, 1e-12)
        cnt["ngrad"] += 1
        gradient = problem.dvector.dual(gmu) + B(step)

        accepted = False
        for _ in range(80):
            trial_y = prox_in_tr_ball(
                y - t * gradient, t, x, radius, problem, cnt
            )
            displacement = trial_y - y
            length = problem.pvector.norm(displacement)
            trial_step = trial_y - x
            trial_value = increment(trial_step)

            slack = 1e-14 * max(1.0, abs(value))
            if (
                math.isfinite(trial_value)
                and trial_value
                <= value - 0.25 * length**2 / t + slack
            ):
                accepted = True
                break

            t *= 0.5

        if not accepted:
            iflag = 2      # Backtracking failed.
            break

        step, value = trial_step, trial_value

        if value <= best_value:
            best_step = step.copy()
            best_value = value

        if length / t <= tol:
            iflag = 0      # Local inner stopping test only.
            break

        t = min(2.0 * t, 1.0)

    s = best_step
    snorm = problem.pvector.norm(s)
    pRed = -best_value
    phinew = problem.obj_nonsmooth.value(x + s)
    cnt["nobj2"] += 1

    return s, snorm, pRed, phinew, iflag, iter_count, cnt, params
