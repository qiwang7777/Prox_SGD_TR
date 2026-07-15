"""
Consistent generalized-gradient / proximal-FCD step selection.
The inner iteration is

    J_r  ->  d_r = P_k(J_r)  ->  J_{r+1} in partial f(x_k + d_r),

and returns the same pair (J_r, d_r) once it has:
  1. positive predicted reduction, and
  2. sufficiently small linearization residual

      |f(x+d_r)-f(x)-<J_r,d_r>| <= tol * ||d_r||.

This directly tests the consistency needed by the outer trust-region ratio.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .subsolver.trustregion_step_DOGLEG import trustregion_step_DOGLEG
from .subsolver.trustregion_step_NCG import trustregion_step_NCG
from .subsolver.trustregion_step_SPG2 import trustregion_step_SPG2
from .subsolver.trustregion_step_SSN import trustregion_step_SSN


@dataclass
class PairInfo:
    success: bool
    inner_iterations: int
    predicted_reduction: float
    linearization_residual: float
    relative_residual: float
    step_norm: float
    subproblem_flag: int
    subproblem_iterations: int
    reason: str


def _copy_vector(x: Any) -> Any:
    return x.copy() if hasattr(x, "copy") else copy.deepcopy(x)


def _smooth_value(obj: Any, x: Any, tol: float) -> float:
    value, _ = obj.value(x, tol)
    return float(value)


def _generalized_gradient(
    x: Any,
    problem: Any,
    params: Dict[str, Any],
    cnt: Dict[str, Any],
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Select one generalized gradient at x.

    This adapter uses obj_smooth.gradient(x, tol). If the application has
    several Clarke selections, replace this function by an active-set or
    branch-selection oracle.
    """
    grad_tol = float(params.get("pair_grad_tol", params.get("gradTol", 1e-12)))
    grad, _ = problem.obj_smooth.gradient(x, grad_tol)
    cnt["ngrad"] = cnt.get("ngrad", 0) + 1
    dgrad = problem.dvector.dual(grad)
    return grad, dgrad, cnt


def _dual_pairing(problem: Any, primal_grad: Any, primal_step: Any) -> float:
    """
    Compute <J,d>.
    """
    return float(problem.pvector.dot(primal_grad, primal_step))


def _solve_model_step(
    x: Any,
    val_model: float,
    grad: Any,
    dgrad: Any,
    phi: float,
    problem: Any,
    params: Dict[str, Any],
    cnt: Dict[str, Any],
):
    solver = str(params.get("spsolver", "SPG2")).upper()

    if solver == "SPG2":
        return trustregion_step_SPG2(
            x, val_model, grad, dgrad, phi, problem, params, cnt
        )
    if solver == "NCG":
        return trustregion_step_NCG(
            x, val_model, dgrad, phi, problem, params, cnt
        )
    if solver == "SSN":
        return trustregion_step_SSN(
            x, val_model, dgrad, phi, problem, params, cnt
        )
    if solver == "DOGLEG":
        return trustregion_step_DOGLEG(
            x, val_model, dgrad, phi, problem, params, cnt
        )

    raise ValueError(f"Unknown trust-region subproblem solver: {solver}")


def select_consistent_pair(
    x: Any,
    val_at_x: float,
    phi_at_x: float,
    problem: Any,
    params: Dict[str, Any],
    cnt: Dict[str, Any],
    *,
    initial_grad: Optional[Any] = None,
    initial_dgrad: Optional[Any] = None,
):
    """
    Compute an approximately consistent pair (J_k, d_k).

    Parameters
    ----------
    x
        Current outer iterate.
    val_at_x
        True smooth/semismooth value f(x).
    phi_at_x
        Nonsmooth convex value phi(x).
    problem, params, cnt
        Objects used by the existing repository.
    initial_grad, initial_dgrad
        Optional initial Clarke selection at x and its dual representation.

    Returns
    -------
    s, snorm, pRed, phinew, iflag, iter_count, cnt, params, grad, dgrad, info

    Notes
    -----
    The routine does not claim that semismoothness alone guarantees finite
    termination. It enforces a maximum number of inner iterations and reports
    failure cleanly so the outer TR method can shrink delta and retry.
    """
    params.setdefault("pair_maxit", 20)
    params.setdefault("pair_tol", 1e-3)
    params.setdefault("pair_abs_tol", 1e-12)
    params.setdefault("pair_pred_tol", 0.0)
    params.setdefault("pair_update_at_endpoint", True)
    params.setdefault("pair_accept_best", False)
    params.setdefault("pair_grad_tol", 1e-12)

    maxit = int(params["pair_maxit"])
    rel_tol = float(params["pair_tol"])
    abs_tol = float(params["pair_abs_tol"])
    pred_tol = float(params["pair_pred_tol"])

    obj = problem.obj_smooth

    if initial_grad is None or initial_dgrad is None:
        grad_r, dgrad_r, cnt = _generalized_gradient(x, problem, params, cnt)
    else:
        grad_r = _copy_vector(initial_grad)
        dgrad_r = _copy_vector(initial_dgrad)

    best = None

    for r in range(maxit):
        # The model value at zero should use f(x), not f at the probe point.
        val_model = float(val_at_x)

        (
            s_r,
            snorm_r,
            pred_r,
            phinew_r,
            iflag_r,
            iter_r,
            cnt,
            params,
        ) = _solve_model_step(
            x,
            val_model,
            grad_r,
            dgrad_r,
            phi_at_x,
            problem,
            params,
            cnt,
        )

        pred_r = float(pred_r)
        snorm_r = float(snorm_r)
        x_trial = x + s_r

        f_trial = _smooth_value(obj, x_trial, float(params["pair_grad_tol"]))
        cnt["nobj1"] = cnt.get("nobj1", 0) + 1

        linear_part = _dual_pairing(problem, grad_r, s_r)
        residual = abs(f_trial - float(val_at_x) - linear_part)
        relative_residual = residual / max(snorm_r, abs_tol)

        info_r = PairInfo(
            success=False,
            inner_iterations=r + 1,
            predicted_reduction=pred_r,
            linearization_residual=residual,
            relative_residual=relative_residual,
            step_norm=snorm_r,
            subproblem_flag=int(iflag_r),
            subproblem_iterations=int(iter_r),
            reason="not accepted",
        )

        score = (relative_residual, -pred_r)
        if best is None or score < best[0]:
            best = (
                score,
                s_r,
                snorm_r,
                pred_r,
                phinew_r,
                iflag_r,
                iter_r,
                _copy_vector(grad_r),
                _copy_vector(dgrad_r),
                info_r,
            )

        pred_ok = pred_r > pred_tol
        consistency_ok = residual <= abs_tol + rel_tol * snorm_r

        if pred_ok and consistency_ok:
            info_r.success = True
            info_r.reason = "positive prediction and consistency tolerance met"
            cnt["pair_inner_iterations"] = cnt.get("pair_inner_iterations", 0) + r + 1
            cnt["pair_last_residual"] = residual
            cnt["pair_last_relative_residual"] = relative_residual

            return (
                s_r,
                snorm_r,
                pred_r,
                phinew_r,
                iflag_r,
                iter_r,
                cnt,
                params,
                grad_r,
                dgrad_r,
                info_r,
            )

        # Update J at the endpoint generated by the current J.
        # This is the set-valued Picard step:
        #     J_{r+1} in partial f(x + P(J_r)).
        if bool(params["pair_update_at_endpoint"]):
            grad_r, dgrad_r, cnt = _generalized_gradient(
                x_trial, problem, params, cnt
            )
        else:
            raise RuntimeError(
                "Only endpoint generalized-gradient updates are implemented."
            )

    cnt["pair_failures"] = cnt.get("pair_failures", 0) + 1

    assert best is not None
    (
        _,
        s_b,
        snorm_b,
        pred_b,
        phinew_b,
        iflag_b,
        iter_b,
        grad_b,
        dgrad_b,
        info_b,
    ) = best

    info_b.reason = "maximum inner iterations reached"

    if bool(params["pair_accept_best"]) and pred_b > pred_tol:
        info_b.success = True
        info_b.reason += "; returning best positive-prediction pair"
        return (
            s_b,
            snorm_b,
            pred_b,
            phinew_b,
            iflag_b,
            iter_b,
            cnt,
            params,
            grad_b,
            dgrad_b,
            info_b,
        )

    # Outer TR code should shrink delta and retry without evaluating rho.
    info_b.success = False
    return (
        s_b,
        snorm_b,
        pred_b,
        phinew_b,
        iflag_b,
        iter_b,
        cnt,
        params,
        grad_b,
        dgrad_b,
        info_b,
    )
