import torch
from semismooth_TR.tr_smoothing_robust import trustregion
import numpy as np
from .setup_helper_smoothing import show_results, build_l1_saturation_problem_smoothing,  build_tv_saturation_problem_smoothing, build_wavelet_l1_saturation_problem_smoothing,check_clipping_margin, check_formula_interior, check_smooth_derivatives, plot_tr_performance, make_interior_test_point
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cpu"

    # A stable first run:
    #n = 64 
    #sigma = 0.25
    #kernel_size = 3
    #noise_level = 0.015 #0.025 will lead to objective stagnation
    #lam = 5.88e-3
    # for making the plot in higher fidelity 
    n=256
    sigma = 1
    kernel_size=7
    noise_level=0.02
    lam = 6e-3
    

    #x_true, b, x0, problem = build_tv_saturation_problem_smoothing(
    #    n=n,
    #    sigma=sigma,
    #    kernel_size=kernel_size,
    #    noise_level=noise_level,
    #    lam=lam,
    #    device=device,
    #    x0_mode="data",
    #)
    
    #x_true, b, x0, problem = build_l1_saturation_problem_smoothing(
    #    n = n,
    #    sigma = sigma,
    #    kernel_size = kernel_size,
    #    noise_level = noise_level,
    #    lam = lam,
    #    device = device,
    #    x0_mode = "data",
    #    )
    x_true, b, x0, problem = build_wavelet_l1_saturation_problem_smoothing(
        n = n,
        sigma = sigma,
        kernel_size = kernel_size,
        noise_level = noise_level,
        lam = lam,
        device = device,
        x0_mode = "data",
        wavelet = "coif3",#"haar",#"db1", #"db2",
        level = 3,    
        )

    params = {
        "spsolver": "SPG2",      # "NCG" or "SPG2"
        "useGCP": True,
        "maxit": 5000,
        "delta": 1.0,
        "gtol": 1e-7,#with n=256, "gtol": 1e-4
        "ocScale": 1.0,
        "eta1": 1e-4,
        "eta2": 0.5,
        "gamma1": 0.5,
        "gamma2": 1.5,
        "maxitsp": 20, #50
        "pred_abs_tol": 1e-12, #1e-10
        "pred_rel_tol": 1e-12, #1e-8
        "pred_small_max": 5, #20
        "outFreq": 1,
        "useInexactGrad": False,
        #smoothing controls
        "use_smoothing_at_deltamin":True,
        "smooth_mode": False,
        "deltamin":1e-4,
        "delta_smooth_exit":5e-4,
        "mu_smooth":1e-3,
        "mu_min":1e-10,
        "mu_factor":1.0,
        "mu_power":1.0,

    }
    
    x_test = make_interior_test_point(n=n,device=device)
    check_smooth_derivatives(problem,x_test,key="img",seed=0)
    check_clipping_margin(problem,x_test)
    check_formula_interior(problem,x_test)

    x_opt, cnt, x_best = trustregion(x0, Deltai=1.0, problem=problem, params=params)
    plot_tr_performance(cnt)

    print("\nFinal status flag:", cnt["iflag"])
    print("Iterations:", cnt["iter"])
    print("Final objective:", cnt["objhist"][-1])
    print("Final relative L2 error:", problem.obj_smooth.relative_L2_error(x_opt))

    show_results(x_true, b, x_best)
