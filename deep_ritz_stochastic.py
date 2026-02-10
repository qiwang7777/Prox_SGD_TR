import numpy as np 
import math, torch,  time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
#import writeSolution


def writeRow(list,file):
    for i in list: file.write("%s "%i)
    file.write("\n")

def write(X,Y,Z,nSampling,file):
    for k1 in range(nSampling):
        writeRow(X[k1],file)
        writeRow(Y[k1],file)
        writeRow(Z[k1],file)

def writeBoundary(edgeList,edgeList2 = None):
    length=[]
    file=open("boundaryCoord.txt","w")

    for i in edgeList:
        writeRow(i,file)
    if edgeList2 != None:
        for i in edgeList2:
            writeRow(i,file)

    file=open("boundaryNumber.txt","w")
    if edgeList2 == None: length = [len(edgeList)]
    else: length = [len(edgeList),len(edgeList2)]

    for i in length:
        file.write("%s\n"%i)

if __name__=="__main__":
    pass


# Sample points in a disk
def sampleFromDisk(r,n):
    """
    r -- radius;
    n -- number of samples.
    """
    array = np.random.rand(2*n,2)*2*r-r
    
    array = np.multiply(array.T,(np.linalg.norm(array,2,axis=1)<r)).T
    array = array[~np.all(array==0, axis=1)]
    
    if np.shape(array)[0]>=n:
        return array[0:n]
    else:
        return sampleFromDisk(r,n)

def sampleFromDomain(n):
    # For simplicity, consider a square with a hole.
    # Square: [-1,1]*[-1,1]
    # Hole: c = (0.3,0.0), r = 0.3
    array = np.zeros([n,2])
    c = np.array([0.3,0.0])
    r = 0.3

    for i in range(n):
        array[i] = randomPoint(c,r)

    return array

def randomPoint(c,r):
    point = np.random.rand(2)*2-1
    if np.linalg.norm(point-c)<r:
        return randomPoint(c,r)
    else:
        return point

def sampleFromBoundary(n):
    # For simplicity, consider a square with a hole.
    # Square: [-1,1]*[-1,1]
    # Hole: c = (0.3,0.0), r = 0.3
    c = np.array([0.3,0.0])
    r = 0.3
    length = 4*2+2*math.pi*r
    interval1 = np.array([0.0,2.0/length])
    interval2 = np.array([2.0/length,4.0/length])
    interval3 = np.array([4.0/length,6.0/length])
    interval4 = np.array([6.0/length,8.0/length])
    interval5 = np.array([8.0/length,1.0])

    array = np.zeros([n,2])

    for i in range(n):
        rand0 = np.random.rand()
        rand1 = np.random.rand()

        point1 = np.array([rand1*2.0-1.0,-1.0])
        point2 = np.array([rand1*2.0-1.0,+1.0])
        point3 = np.array([-1.0,rand1*2.0-1.0])
        point4 = np.array([+1.0,rand1*2.0-1.0])
        point5 = np.array([c[0]+r*math.cos(2*math.pi*rand1),c[1]+r*math.sin(2*math.pi*rand1)])

        array[i] = myFun(rand0,interval1)*point1 + myFun(rand0,interval2)*point2 + \
            myFun(rand0,interval3)*point3 + myFun(rand0,interval4)*point4 + \
                myFun(rand0,interval5)*point5
 
    return array

def myFun(x,interval):
    if interval[0] <= x <= interval[1]:
        return 1.0
    else: return 0.0

def sampleFromSurface(r,n):
    """
    r -- radius;
    n -- number of samples.
    """
    array = np.random.normal(size=(n,2))
    norm = np.linalg.norm(array,2,axis=1)
    # print(np.min(norm))
    if np.min(norm) == 0:
        return sampleFromSurface(r,n)
    else:
        array = np.multiply(array.T,1/norm).T
        return array*r



if __name__ == "__main__":
    # array = sampleFromDomain(10000).T
    # array = sampleFromBoundary(500).T
    # plt.plot(array[0],array[1],'o',ls="None")
    # plt.axis("equal")
    # plt.show()
    pass
# Network structure
class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList([nn.Linear(params["width"],params["width"]) for _ in range(params["depth"])])
        

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.linearIn(x)) # Match dimension
        for layer in self.linear:
            x_temp = self.act(layer(x))
            x = x_temp
        
        return self.linearOut(x)
    
class HardBCNet(nn.Module):
    def __init__(self, base_net, radius):
        super().__init__()
        self.base = base_net
        self.r = radius

    def forward(self, x):
        # x: (N,2)
        b = self.r**2 - torch.sum(x*x, dim=1, keepdim=True)  # (N,1)
        return b * self.base(x)  # g(x)=0 for your problem


@torch.no_grad()
def prox_grad_step(model, lr, beta, exclude_bias=False):
    """
    One proximal-gradient update on all parameters:
      θ <- prox_{lr*beta*||.||_1}( θ - lr * grad )
    """
    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        if exclude_bias and ("bias" in name):
            # optional: don't L1-penalize biases
            p.add_(p.grad, alpha=-lr)
            continue

        # gradient step on smooth part
        p.add_(p.grad, alpha=-lr)

        # proximal step for L1
        soft_threshold_(p, lr * beta)

def train_prox(model, device, params):
    model.train()

    data1 = torch.from_numpy(sampleFromDisk(params["radius"], params["bodyBatch"])).float().to(device)
    data1.requires_grad = True
    data2 = torch.from_numpy(sampleFromSurface(params["radius"], params["bdryBatch"])).float().to(device)

    lr0 = params["lr"]
    step_size = params["step_size"]
    gamma = params["gamma"]

    for step in range(params["trainStep"] - params["preStep"]):

        # manual StepLR-like schedule
        lr = lr0 * (gamma ** (step // step_size))

        output1 = model(data1)
        model.zero_grad(set_to_none=True)

        dfdx = torch.autograd.grad(
            output1, data1,
            grad_outputs=torch.ones_like(output1),
            retain_graph=True, create_graph=True, only_inputs=True
        )[0]

        fTerm = ffun(data1).to(device)

        loss1 = torch.mean(0.5 * torch.sum(dfdx * dfdx, 1).unsqueeze(1) - fTerm * output1) \
                * math.pi * params["radius"]**2

        output2 = model(data2)
        target2 = exact(params["radius"], data2)
        loss2 = torch.mean((output2 - target2) ** 2 * params["penalty"] * 2 * math.pi * params["radius"])

        # smooth part ONLY
        loss_smooth = loss1 + loss2

        if step % params["writeStep"] == params["writeStep"] - 1:
            with torch.no_grad():
                target = exact(params["radius"], data1)
                error = errorFun(output1, target, params)
                print(f"Error at Step {step + params['preStep'] + 1} is {error}")
            with open("lossData.txt", "a") as file:
                file.write(f"{step + params['preStep'] + 1} {error}\n")

        if step % params["sampleStep"] == params["sampleStep"] - 1:
            data1 = torch.from_numpy(sampleFromDisk(params["radius"], params["bodyBatch"])).float().to(device)
            data1.requires_grad = True
            data2 = torch.from_numpy(sampleFromSurface(params["radius"], params["bdryBatch"])).float().to(device)

        if 10 * (step + 1) % params["trainStep"] == 0:
            print(f"{100 * (step + 1) // params['trainStep']}% finished...")

        loss_smooth.backward()

        # proximal gradient update (ISTA)
        prox_grad_step(model, lr=lr, beta=params["beta"], exclude_bias=False)
        
def build_l1_mask(model, exclude_bias=True):
    mask_list = []
    for name, p in model.named_parameters():
        m = torch.ones(p.numel(), dtype=torch.float32)
        if exclude_bias and ("bias" in name):
            m.zero_()
        mask_list.append(m)
    return torch.cat(mask_list)

def soft_threshold_vec_masked(v, thresh, mask):
    # apply soft-threshold only where mask==1
    out = v.clone()
    idx = mask.bool()
    out[idx] = torch.sign(v[idx]) * torch.clamp(v[idx].abs() - thresh, min=0.0)
    return out


def soft_threshold_(p, thresh):
    p.sign_().mul_(torch.clamp(p.abs_().sub_(thresh), min=0.0))

@torch.no_grad()
def prox_sgd_step(model, lr, beta, exclude_bias=True):
    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        # plain SGD step on smooth part
        p.add_(p.grad, alpha=-lr)

        # prox for L1 (usually don't penalize biases)
        if exclude_bias and ("bias" in name):
            continue
        soft_threshold_(p, lr * beta)


def train_prox_sgd(model, device, params):
    error_history = []
    model.train()

    lr0 = params["lr"]
    step_size = params["step_size"]
    gamma = params["gamma"]

    for step in range(params["trainStep"] - params["preStep"]):

        # --- PROX-SGD: resample each iteration ---
        data1 = torch.from_numpy(sampleFromDisk(params["radius"], params["bodyBatch"])).float().to(device)
        data1.requires_grad_(True)
        data2 = torch.from_numpy(sampleFromSurface(params["radius"], params["bdryBatch"])).float().to(device)

        # StepLR-like schedule (optional)
        lr = lr0 * (gamma ** (step // step_size))

        # forward
        output1 = model(data1)
        model.zero_grad(set_to_none=True)

        dfdx = torch.autograd.grad(
            output1, data1,
            grad_outputs=torch.ones_like(output1),
            retain_graph=True, create_graph=True
        )[0]

        fTerm = ffun(data1).to(device)

        loss1 = torch.mean(
            0.5 * torch.sum(dfdx * dfdx, dim=1, keepdim=True) - fTerm * output1
        ) * math.pi * params["radius"]**2

        output2 = model(data2)
        target2 = exact(params["radius"], data2)
        loss2 = torch.mean((output2 - target2) ** 2) * params["penalty"] * (2 * math.pi * params["radius"])

        loss_smooth = loss1 + loss2
        loss_smooth.backward()

        # (optional) gradient clipping for stability
        if "grad_clip" in params and params["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])

        # prox-SGD update
        prox_sgd_step(model, lr=lr, beta=params["beta"], exclude_bias=True)

        # logging
        if step % params["writeStep"] == params["writeStep"] - 1:
            with torch.no_grad():
                target = exact(params["radius"], data1)
                err = errorFun(output1, target, params)

            error_history.append(err)
            print(f"Error at Step {step + params['preStep'] + 1} is {err}")

            with open("lossData.txt", "a") as f:
                f.write(f"{step + params['preStep'] + 1} {err}\n")


        if 10 * (step + 1) % params["trainStep"] == 0:
            print(f"{100 * (step + 1) // params['trainStep']}% finished...")
    return error_history



    


    

def errorFun(output,target,params):
    error = output-target
    error = math.sqrt(torch.mean(error*error)*math.pi*params["radius"]**2)
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target)*math.pi*params["radius"]**2)
    return error/ref   

def test(model,device,params):
    numQuad = params["numQuad"]

    data = torch.from_numpy(sampleFromDisk(1,numQuad)).float().to(device)
    output = model(data)
    target = exact(params["radius"],data).to(device)

    error = output-target
    error = math.sqrt(torch.mean(error*error)*math.pi*params["radius"]**2)
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target)*math.pi*params["radius"]**2)
    return error/ref

def ffun(data):
    # f = 4
    return 4.0*torch.ones([data.shape[0],1],dtype=torch.float)

def exact(r,data):
    # f = 4 ==> u = r^2-x^2-y^2
    output = r**2-torch.sum(data*data,dim=1)

    return output.unsqueeze(1)

def rough(r,data):
    # A rough guess
    output = r**2-r*torch.sum(data*data,dim=1)**0.5
    return output.unsqueeze(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


import copy

def flatten_params(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])

def flatten_grads(model):
    gs = []
    for p in model.parameters():
        if p.grad is None:
            gs.append(torch.zeros_like(p).view(-1))
        else:
            gs.append(p.grad.detach().view(-1))
    return torch.cat(gs)

@torch.no_grad()
def add_step_to_model(model, step_vec):
    # step_vec is flat; add into parameters
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.add_(step_vec[offset:offset+n].view_as(p))
        offset += n

@torch.no_grad()
def set_model_from_flat(model, flat_vec):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.copy_(flat_vec[offset:offset+n].view_as(p))
        offset += n

@torch.no_grad()
def l1_norm_params(model, exclude_bias=True):
    s = 0.0
    for name, p in model.named_parameters():
        if exclude_bias and ("bias" in name):
            continue
        s += p.abs().sum().item()
    return s

def soft_threshold_vec(v, thresh):
    return torch.sign(v) * torch.clamp(v.abs() - thresh, min=0.0)

def smooth_loss_and_grad(model, device, params, data1, data2):
    model.train()
    data1 = data1.to(device)
    data1.requires_grad_(True)
    data2 = data2.to(device)

    output1 = model(data1)
    model.zero_grad(set_to_none=True)

    dfdx = torch.autograd.grad(
        output1, data1,
        grad_outputs=torch.ones_like(output1),
        retain_graph=True, create_graph=True
    )[0]

    fTerm = ffun(data1).to(device)

    loss1 = torch.mean(
        0.5 * torch.sum(dfdx * dfdx, dim=1, keepdim=True) - fTerm * output1
    ) * math.pi * params["radius"]**2

    output2 = model(data2)
    target2 = exact(params["radius"], data2)
    loss2 = torch.mean((output2 - target2) ** 2) * params["penalty"] * (2 * math.pi * params["radius"])

    loss_smooth = loss1 + loss2
    loss_smooth.backward()

    g = flatten_grads(model)
    return loss_smooth.detach().item(), g.detach()


def total_objective(model, device, params, data1, data2, exclude_bias=True):
    # smooth part (no grad)
    model.eval()
    data1 = data1.to(device)
    data1.requires_grad_(True)
    data2 = data2.to(device)

    output1 = model(data1)
    dfdx = torch.autograd.grad(
        output1, data1,
        grad_outputs=torch.ones_like(output1),
        retain_graph=True, create_graph=False
    )[0]

    fTerm = ffun(data1).to(device)
    loss1 = torch.mean(
        0.5 * torch.sum(dfdx * dfdx, dim=1, keepdim=True) - fTerm * output1
    ) * math.pi * params["radius"]**2

    output2 = model(data2)
    target2 = exact(params["radius"], data2)
    loss2 = torch.mean((output2 - target2) ** 2) * params["penalty"] * (2 * math.pi * params["radius"])

    l1 = params["beta"] * l1_norm_params(model, exclude_bias=exclude_bias)
    return (loss1 + loss2).item() + l1


def propose_step_prox_sgd(model, device, params, Delta, mask, inner_steps=5, lr=1e-3, exclude_bias=True):
    theta0 = flatten_params(model)
    theta = theta0.clone().to(device)

    for _ in range(inner_steps):
        # stochastic batch each inner step
        data1 = torch.from_numpy(sampleFromDisk(params["radius"], params["bodyBatch"])).float()
        data2 = torch.from_numpy(sampleFromSurface(params["radius"], params["bdryBatch"])).float()

        # set model to current theta
        set_model_from_flat(model, theta)

        _, g = smooth_loss_and_grad(model, device, params, data1, data2)

        # gradient step
        theta = theta - lr * g

        # prox on L1 (in flat space)
        # if excluding bias, you'd need a mask; simplest: include all params in prox OR skip bias in model-space prox.
        # Here: apply prox to all params for simplicity:
        theta = soft_threshold_vec_masked(theta, lr * params["beta"],mask)

    s = theta - theta0

    # Trust-region projection: clip step to ||s|| <= Delta
    snorm = torch.norm(s).item()
    if snorm > Delta:
        s = s * (Delta / (snorm + 1e-12))

    return s


def trust_region_prox_sgd(model, device, params):
    # --- TR parameters ---
    Delta = params.get("Delta0", 1e-2)
    Delta_max = params.get("Delta_max", 1.0)
    Delta_min = params.get("Delta_min", 1e-6)

    eta1 = params.get("eta1", 0.1)     # reject threshold
    eta2 = params.get("eta2", 0.75)    # expand threshold
    gamma_dec = params.get("gamma_dec", 0.5)
    gamma_inc = params.get("gamma_inc", 1.5)

    outer_iters = params.get("outer_iters", 50)
    inner_steps = params.get("inner_steps", 5)
    lr = params.get("lr_theta", 1e-3)  # <-- t = lr for prox residual

    # --- termination threshold: ||x - prox|| ---
    tol_gmap = params.get("tol_gmap", 1.0)     # scaled by lr
    need_consec = params.get("need_consec", 3)

    # --- build L1 mask once (exclude bias typical) ---
    exclude_bias = params.get("exclude_bias", True)
    mask = build_l1_mask(model, exclude_bias=exclude_bias).to(device)

    history = {"F": [], "Delta": [], "rho": [], "xprox": [], "gmap": []}
    good_count = 0

    for k in range(outer_iters):
        theta0 = flatten_params(model).to(device)

        # evaluation batch (you can make these larger to reduce noise)
        eval_body = params.get("eval_bodyBatch", params["bodyBatch"])
        eval_bdry = params.get("eval_bdryBatch", params["bdryBatch"])
        data1e = torch.from_numpy(sampleFromDisk(params["radius"], eval_body)).float()
        data2e = torch.from_numpy(sampleFromSurface(params["radius"], eval_bdry)).float()

        # compute smooth loss grad on eval batch
        set_model_from_flat(model, theta0.detach().cpu())
        f0_smooth, g0 = smooth_loss_and_grad(model, device, params, data1e, data2e)
        g0 = g0.to(device)

        # total objective (smooth + beta*L1 with same exclude_bias)
        F0 = total_objective(model, device, params, data1e, data2e, exclude_bias=exclude_bias)

        # --- prox residual with t = lr ---
        with torch.no_grad():
            
            z = theta0 - lr * g0
            proxz = soft_threshold_vec_masked(z, lr * params["beta"], mask)

            xprox = torch.norm(theta0 - proxz).item()  # termination metric
            #tol_gmap = params.get("tol_gmap", 1e-2)     # choose what “stationary enough” means
            #tol_xprox = lr * tol_gmap
            gmap  = xprox / (lr + 1e-16)                # optional for logging

        history["F"].append(F0)
        history["Delta"].append(Delta)
        history["xprox"].append(xprox)
        history["gmap"].append(gmap)

        # --- termination ---
        tol_xprox = lr * tol_gmap

        if xprox < tol_xprox:
            good_count += 1
        else:
            good_count = 0

        if good_count >= need_consec:
            print(
                f"[TR] Stop: xprox {xprox:.3e} < lr*tol_gmap {tol_xprox:.3e} "
                f"for {need_consec} consecutive iterations "
                f"(lr={lr:.1e}, tol_gmap={tol_gmap:.1e})."
            )
            break

        if Delta < Delta_min:
            print(f"[TR] Stop: Delta {Delta:.3e} < {Delta_min:.3e}")
            break

        #if xprox < tol_xprox:
        #    print(f"[TR] Stop: xprox {xprox:.3e} < {tol_xprox:.3e} at outer {k}")
        #    break
        #if Delta < Delta_min:
        #    print(f"[TR] Stop: Delta {Delta:.3e} < {Delta_min:.3e} at outer {k}")
        #    break

        # --- propose a step (must use same mask/prox rule) ---
        s = propose_step_prox_sgd(
            model, device, params, Delta,
            inner_steps=inner_steps, lr=lr,
            mask=mask, exclude_bias=exclude_bias
        ).to(device)

        snorm = torch.norm(s).item()
        if snorm < 1e-14:
            print("[TR] Step nearly zero; shrinking Delta.")
            Delta *= gamma_dec
            continue

        # --- predicted reduction: linear smooth + exact masked L1 change ---
        with torch.no_grad():
            theta_trial = theta0 + s

            l1_0 = torch.sum(torch.abs(theta0) * mask).item()
            l1_1 = torch.sum(torch.abs(theta_trial) * mask).item()

            pred = -(g0 @ s).item() + params["beta"] * (l1_0 - l1_1)
            pred = max(pred, 1e-12)

        # --- actual reduction on same eval batch ---
        set_model_from_flat(model, theta_trial.detach().cpu())
        F1 = total_objective(model, device, params, data1e, data2e, exclude_bias=exclude_bias)

        ared = F0 - F1
        rho = ared / pred
        history["rho"].append(rho)

        # --- accept/reject + update radius ---
        if rho < eta1 or ared <= 0:
            # reject
            set_model_from_flat(model, theta0.detach().cpu())
            Delta *= gamma_dec
            accepted = False
        else:
            # accept
            accepted = True
            if rho > eta2:
                Delta = min(Delta_max, gamma_inc * Delta)

        print(
            f"[TR] outer {k:03d} | F {F0:.4e} -> {F1:.4e} | ared {ared:.3e} | pred {pred:.3e} | "
            f"rho {rho:.2f} | Delta {Delta:.2e} | acc {accepted} | xprox {xprox:.2e} | gmap {gmap:.2e}"
        )

    return history




def main():
    # Parameters
    # torch.manual_seed(21)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()

    #Problem/geometry
    params["radius"] = 1
    params["d"] = 2 # 2D
    params["dd"] = 1 # Scalar field
    #Sampling
    params["bodyBatch"] = 4096 # Batch size
    params["bdryBatch"] = 4096 # Batch size for the boundary integral
    params["numQuad"] = 40000 # Number of quadrature points for testing
    #Network
    params["width"] = 8 # Width of layers
    params["depth"] = 2 # Depth of the network: depth+2
    #Optimization
    params["inner_steps"] = 20 # prox-SGD steps per TR subproblem
    params["lr_theta"] = 2e-4 # stepsize t for prox-gradient (USED)
    params["beta"] = 1e-6 # L1 regularization weight
    #TR    
    params["Delta0"] = 1e-1
    params["Delta_min"] = 1e-5
    params["Delta_max"] = 1.0
    params["gamma_inc"] = 1.5
    params["gamma_dec"] = 0.5
    #Termination
    params["outer_iters"] = 200
    params["tol_gmap"] = 0.3 # scaled: tol_xprox = lr_theta * tol_gmap
    params["need_consec"] = 5 # consecutive satisfaction required
    #Logging/plotting
    params["writeStep"] = 1
    # PDE loss
    params["penalty"] = 500       # boundary penalty weight
    #Stability
    params["grad_clip"] = 1.0 # gradient clipping in inner solves


    startTime = time.time()
    base = RitzNet(params).to(device)
    model = HardBCNet(base, params["radius"]).to(device)
    print("Generating network costs %s seconds."%(time.time()-startTime))

    #preOptimizer = torch.optim.Adam(model.parameters(),lr=params["preLr"])
    #optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    #scheduler = StepLR(optimizer,step_size=params["step_size"],gamma=params["gamma"])
    

    startTime = time.time()
    #preTrain(model,device,params,preOptimizer,None,rough)
    #error_history = train_prox_sgd(model,device,params)
    #train_admm(model,device,params)
    #trust_region_admm(model, device, params)
    history = trust_region_prox_sgd(model, device, params)
    
    # plot xprox vs outer iteration
    plt.figure()
    plt.plot(history["xprox"])
    plt.yscale("log")
    plt.xlabel("Outer iteration")
    plt.ylabel(r"$\|x - \mathrm{prox}(x - t\nabla f)\|$")
    plt.title("Proximal stationarity (xprox)") 
    plt.grid(True, which="both", ls="--")
    plt.show()
    
    #gmap vs outer iteration
    plt.figure()
    plt.plot(history["gmap"])
    plt.yscale("log")
    plt.xlabel("Outer iteration")
    plt.ylabel("Prox-gradient mapping norm")
    plt.title("Composite stationarity (gmap)")
    plt.grid(True, which="both", ls="--")
    plt.show()



    # plot convergence
    plt.figure()
    plt.plot(history["F"])
    #plt.yscale("log")
    plt.xlabel("Outer iteration")
    plt.ylabel("Objective F")
    plt.title("TR + prox-SGD convergence")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()
    testError = test(model,device,params)
    print("The test error (of the last model) is %s."%testError)
    print("The number of parameters is %s "%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")

    #pltResult(model, device, 200, params, exact_fun=lambda xx,yy: exact_fun_lshape(xx,yy,a=params["a"]))
    pltResult(model, device, 200, params)
    #plot_convergence(error_history, params)

    

def plot_convergence(error_history, params):
    steps = np.arange(1, len(error_history) + 1) * params["writeStep"]

    plt.figure(figsize=(6,4))
    plt.plot(steps, error_history, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Relative L2 error")
    plt.title("Prox-SGD convergence")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


    




def pltResult(model, device, nSample, params):
    model.eval()
    r = params["radius"]

    # Create a Cartesian grid on [-r, r] x [-r, r]
    x = np.linspace(-r, r, nSample)
    y = np.linspace(-r, r, nSample)
    xx, yy = np.meshgrid(x, y)

    # Flatten grid to feed into NN
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Mask for the disk
    rr2 = xx**2 + yy**2
    mask = rr2 <= r**2

    # Evaluate NN on all grid points
    with torch.no_grad():
        data = torch.from_numpy(points).float().to(device)
        u_nn = model(data).cpu().numpy().reshape(xx.shape)

    # Exact solution: u = r^2 - x^2 - y^2
    u_exact = r**2 - rr2

    # Mask out values outside the disk
    u_nn_masked = np.where(mask, u_nn, np.nan)
    u_exact_masked = np.where(mask, u_exact, np.nan)
    err_masked = u_nn_masked - u_exact_masked

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(
        u_exact_masked,
        extent=[-r, r, -r, r],
        origin="lower",
    )
    axes[0].set_title("Exact solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        u_nn_masked,
        extent=[-r, r, -r, r],
        origin="lower",
    )
    axes[1].set_title("NN approximation")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        err_masked,
        extent=[-r, r, -r, r],
        origin="lower",
    )
    axes[2].set_title("Error (NN - exact)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

    # --- If you still want to write data to files (optional) ---
    # Save nSample
    file = open("nSample.txt","w")
    file.write(str(nSample))
    file.close()

    # Save data (only inside disk if you like, here we save full grids)
    file = open("Data.txt","w")
    write(xx, yy, u_nn, nSample, file)
    file.close()

    # Boundary points (circle)
    thetaList = np.linspace(0, 2 * math.pi, nSample)
    edgeList = [[r * math.cos(t), r * math.sin(t)] for t in thetaList]
    writeBoundary(edgeList)



if __name__=="__main__":
    main()
