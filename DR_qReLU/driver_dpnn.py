from .training_helper import plot_tr_history, plot_solution_and_error,train_poisson_with_TR, vector_from_model, make_training_points_grid, compute_g_from_u_star, kappa_xy
import torch
from .qReLU_NN import PoissonNet
from .Poisson_obj import PoissonCompositeObjective, u_star
from semismooth_TR.derivative_check import grad_check, gn_quadratic_form_check

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    width, depth, ngrid = 16, 2, 64
    beta = 1e-6
    model = PoissonNet(width = width, depth = depth).to(device)
    xy = make_training_points_grid(ngrid, device = device)
    g = compute_g_from_u_star(xy)
    var = {"useEuclidean": False, "beta": beta}
    obj_smooth = PoissonCompositeObjective(model = model, xy= xy, g = g, kappa_fn = kappa_xy, weight = None, device = device, mu_I = 0.0, x_true=True, u_true_fn=u_star)
    obj_smooth.set_hess_mode("full")
    x0 = vector_from_model(model)
    #Derivative check and Hessian check
    print("\n ==== GRAD CHECK at x0 ====")
    grad_check(obj_smooth, x0, ntests=5)
    print("\n ==== HV CHECK at x0 ====")
    gn_quadratic_form_check(obj_smooth, x0, ntests = 3)
    
    torch.autograd.set_detect_anomaly(True)
    

    model, x_opt, cnt,_ = train_poisson_with_TR(
        width=width,
        depth=depth,
        ngrid=ngrid,
        beta=beta,
        delta0=1.0,
        maxit= 1000,
        device=device,
    )
    

    plot_solution_and_error(model, n=121, device=device)
    plot_tr_history(cnt)
