import torch

from .training_helper_obstacle_oc import (
    train_obstacle_oc_with_TR,
    plot_obstacle_oc,
    plot_tr_history,
)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    obj, u_opt, cnt, best_u = train_obstacle_oc_with_TR(
        n=64,
        alpha=1e-4,
        beta=1e-7,
        gamma=1e5,
        delta0=1.0,
        maxit=800,
        device=device,
        mu_smooth=1e-6,
        deltamin=1e-5,
        deltamax=1e4,
    )

    print("Final obstacle violation:", obj.relative_obstacle_violation(u_opt).item())

    plot_obstacle_oc(obj, u_opt)
    plot_tr_history(cnt)
