import numpy as np
import torch
import matplotlib.pyplot as plt
from semismooth_TR.set_default_parameters import set_default_parameters
from semismooth_TR.trust_region import trustregion
from semismooth_TR.derivative_check import grad_check
from .ControlVector import ControlVector
from .L1 import L1Penalty
from .Problem_wrapper import Problem
from .Reduced_obj import ProjectedSemilinearControlObjective
from .sampling import make_interior_grid,desired_state,build_negative_laplacian

torch.set_default_dtype(torch.float64)

def solve_projected_semilinear_control(ngrid=34,alpha=1e-4,beta=1e-7,a=0.0,b=0.4,
                                       delta0=1.0,maxit=100,device="cpu"):
    grid,X,Y,xy=make_interior_grid(ngrid,device=device)
    yd=desired_state(xy); A,h=build_negative_laplacian(ngrid,device=device)
    var={"useEuclidean":True,"beta":beta,"weight":h*h}
    smooth=ProjectedSemilinearControlObjective(A,yd,alpha,a,b,h*h,xy=xy)
    problem=Problem(smooth,L1Penalty(var),var)
    x0=ControlVector(torch.zeros(((ngrid-2)**2,1),device=device))
    params=set_default_parameters("SPG2")
    params["delta"]=delta0; params["maxit"]=maxit; params["gtol"]=1e-7
    params["useInexactObj"]=False; params["useInexactGrad"]=False
    x,cnt,best=trustregion(x0,delta0,problem,params)
    return x,cnt,problem,X,Y,yd

def plot_solution(problem,u,X,Y,yd):
    n=X.shape[0]; y=problem.obj_smooth._state_for(u)
    Xn,Yn=X.cpu().numpy(),Y.cpu().numpy()
    yd=yd.reshape(n,n).cpu().numpy(); yn=y.reshape(n,n).cpu().numpy()
    un=u.data.reshape(n,n).cpu().numpy()
    proj=np.clip(yn,problem.obj_smooth.a,problem.obj_smooth.b)
    fig,ax=plt.subplots(1,4,figsize=(18,4.5),constrained_layout=True)
    for a_,z,title in zip(ax,[yd,yn,un,proj],
        [r"desired state $y_d$",r"computed state $y$",r"sparse control $u$",r"$P_{[a,b]}(y)$"]):
        im=a_.pcolormesh(Xn,Yn,z,shading="auto"); fig.colorbar(im,ax=a_)
        a_.set_title(title); a_.set_xlabel(r"$x_1$"); a_.set_ylabel(r"$x_2$"); a_.set_aspect("equal")
    ax[1].contour(Xn,Yn,yn,levels=[problem.obj_smooth.a,problem.obj_smooth.b])
    plt.show()

def plot_tr_history(cnt):
    obj=np.asarray(cnt.get("objhist",[])); g=np.asarray(cnt.get("gnormhist",[])); d=np.asarray(cnt.get("deltahist",[]))
    fig,ax=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
    ax[0].plot(obj); ax[0].set_title("Objective"); ax[0].set_xlabel("iteration")
    ax[1].plot(g); ax[1].set_yscale("log"); ax[1].set_title("prox-gradient norm"); ax[1].set_xlabel("iteration")
    ax[2].plot(d); ax[2].set_yscale("log"); ax[2].set_title("trust-region radius"); ax[2].set_xlabel("iteration")
    plt.show()

if __name__=="__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    ngrid=34; alpha=1e-4; beta=1e-7; a=0.0; b=0.4
    _,X,Y,xy=make_interior_grid(ngrid,device=device)
    yd=desired_state(xy); A,h=build_negative_laplacian(ngrid,device=device)
    test=ProjectedSemilinearControlObjective(A,yd,alpha,a,b,h*h,xy=xy)
    xt=ControlVector(1e-2*torch.randn(((ngrid-2)**2,1),device=device))
    print("\n==== GENERALIZED GRADIENT CHECK ===="); grad_check(test,xt,ntests=3)
    u,cnt,problem,X,Y,yd=solve_projected_semilinear_control(ngrid,alpha,beta,a,b,1.0,100,device)
    low,mid,high=problem.obj_smooth.active_fractions(u)
    print("\nFinal objective:",cnt["objhist"][-1]); print("Termination flag:",cnt["iflag"])
    print("Lower saturation fraction:",low); print("Interior projection fraction:",mid); print("Upper saturation fraction:",high)
    print("Fraction near-zero controls:",float(torch.mean((torch.abs(u.data)<=1e-8).to(u.data.dtype)).item()))
    plot_solution(problem,u,X,Y,yd); plot_tr_history(cnt)
