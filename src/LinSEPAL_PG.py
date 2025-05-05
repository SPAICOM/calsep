from autograd import jacobian
from autograd import numpy as anp  # Use autograd.numpy
from autograd.numpy.linalg import inv
from pymanopt.manifolds import Stiefel
from autograd.numpy.random import uniform
from scipy.sparse.linalg import cg
from src.utils import f

def soft_thresholding(x, threshold):
    """
    This function implements the element-wise soft-thresholding operator.

    INPUT
    =====
    - x: anp.array, argument.
    - threshold: float/anp.array, threshold.
    
    OUTPUT
    ======
    - anp.array. Soft-thresholded x.
    """
    return anp.sign(x) * anp.maximum(anp.abs(x) - threshold, 0)

def compute_basis_normal_space(V):
    """
    This function computes the basis for the normal space of the Stiefel manifold at V.

    INPUT
    =====
    - V: anp.array, shape (l,h). Point of the manifold.

    OUTPUT
    ======
    - basis: anp.array, shape (s, l, h). Basis for the normal space, where s=h(h+1)/2.
    """

    _, h = V.shape
    basis = []
    for i in range(h):
        for j in range(i, h):
            E = anp.zeros((h, h))
            E[i, j] = E[j, i]=1
            basis.append(V @ E)
    return anp.array(basis)

def grad_f(V, covlow, covhigh):
    """This function computes the gradient w.r.t V of the KL divergence between the high-level SCM probability measure and
    the projection of the low-level SCM probability measure onto R^h.

    INPUT
    =====
    - V: anp.array, shape (l,h). Linear measurable map belonging to the Stiefel manifold.
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.

    OUTPUT
    ======    
    - grad: anp.array, shape (l, h). Gradient of the KL divergence. 
    """
    _, h = V.shape
    grad = 2 * covlow @ V @ inv(V.T @ covlow @ V) @ (-covhigh @ inv(V.T @ covlow @ V) + anp.eye(h))
    return grad

def objective(V, covlow, covhigh, DI, lambda_reg):
    """
    This function implements the objective f(V) + h(V).

    INPUT
    =====
    - V: anp.array, shape (l,h). Linear measurable map belonging to the Stiefel manifold.
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.
    - DI: anp.array, shape (l,h). Penalty matrix.
    - lambda_reg: float>0. ell_1-norm penalty.

    OUTPUT
    ======
    - float. Objective function value
    """
    return f(V, covlow, covhigh) + lambda_reg*anp.linalg.norm(DI*V, ord=1)

def F(mu_k, V_k, DI, basis_normal, B_k, gradf, lambda_reg, rho):
    """
    This function computes F(mu) for the regularized semi-smooth Newton method.
    
    INPUT
    =====
    - mu_k: anp.array, shape (s,). Lagrangian multiplier.
    - V_k: anp.array, shape (l,h). Linear measurable map belonging to the Stiefel manifold at iteration k.
    - DI: anp.array, shape (l,h). Penalty matrix.
    - basis_normal: anp.array, shape (s, l, h). Basis for the normal space at V_k, where s=h(h+1)/2.
    - B_k: anp.array, shape (s, l*h). Flattening of basis_normal obtained by vertically stacking the vectorization of the matrices constituting basis_normal at V_k. 
    - gradf: anp.array, shape (l, h). Gradient of the KL divergence at V_k. 
    - lambda_reg: float>0. ell_1-norm penalty.
    - rho: float. Augmented Lagrangian penalty.
    
    OUTPUT
    ======
    - anp.array, shape (s, s). Evaluation of F computed at mu_k.
    """
    B_mu = V_k - rho * (gradf - anp.tensordot(basis_normal, mu_k, axes=([0], [0])))
    P = soft_thresholding(B_mu, lambda_reg * rho * DI)
    G_mu = P - V_k
    return B_k @ G_mu.flatten()

def J(mu_k, V_k, DI, basis_normal, B_k, gradf, lambda_reg, rho):
    """
    This function computes F(mu) for the regularized semi-smooth Newton method.
    
    INPUT
    =====
    - mu_k: anp.array, shape (s,). Lagrangian multiplier.
    - V_k: anp.array, shape (l,h). Linear measurable map belonging to the Stiefel manifold at iteration k.
    - DI: anp.array, shape (l,h). Penalty matrix.
    - basis_normal: anp.array, shape (s, l, h). Basis for the normal space at V_k, where s=h(h+1)/2.
    - B_k: anp.array, shape (s, l*h). Flattening of basis_normal obtained by vertically stacking the vectorization of the matrices constituting basis_normal at V_k. 
    - gradf: anp.array, shape (l, h). Gradient of the KL divergence at V_k. 
    - lambda_reg: float>0. ell_1-norm penalty.
    - rho: float>0. Augmented Lagrangian penalty.
    
    OUTPUT
    ======
    - anp.array, shape (s, s). Evaluation of the Jacobian of F at mu_k.
    """
    return jacobian(F, argnum=0)(mu_k, V_k, DI, basis_normal, B_k, gradf, lambda_reg, rho)

def safeguard_step(V_k, DI, mu_k, d_k, F_mu_k, basis_normal, B_k, gradf, lambda_reg, rho, alpha_k, beta_k, phi_1, phi_2, psi_1, psi_2, bar_alpha, gamma, delta):
    """
    This function implements the safeguard step for the regularized semi-smooth Newton method.

    INPUT
    =====
    - V_k: anp.array, shape (l,h). Linear measurable map belonging to the Stiefel manifold at iteration k. 
    - DI: anp.array, shape (l,h). Penalty matrix. 
    - mu_k: anp.array, shape (s,). Lagrangian multiplier. 
    - d_k: anp.array, shape (s,). Descent step direction. 
    - F_mu_k: anp.array, shape (s, s). Evaluation of F computed at mu_k. 
    - basis_normal: anp.array, shape (s, l, h). Basis for the normal space at V_k, where s=h(h+1)/2. 
    - B_k: anp.array, shape (s, l*h). Flattening of basis_normal obtained by vertically stacking the vectorization of the matrices constituting basis_normal at V_k. 
    - gradf: anp.array, shape (l, h). Gradient of the KL divergence at V_k.   
    - lambda_reg: float>0. ell_1-norm penalty. 
    - rho: float>0. Augmented Lagrangian penalty. 
    - alpha_k: float>0. Value at iteration k. 
    - beta_k: float>0. Value at iteration k.
    - phi_1: float in (0, phi_2]. 
    - phi_2: float in (0, 1). 
    - psi_1: float in (1, psi_2). 
    - psi_2: float>1.
    - bar_alpha: float>0.
    - gamma: float in (0,1).
    - delta: float in (0, 1/omega).
    
    OUTPUT
    ======
    - mu_k: anp.array, shape (s,). Updated Lagrangian multiplier.
    - alpha_k: float>0. Updated value for iteration k+1. 
    - beta_k: float>0. Updated value for iteration k+1.
    """

    u_k = mu_k + d_k
    F_u_k = F(u_k, V_k, DI, basis_normal, B_k, gradf, lambda_reg, rho)

    norm_F_u_k = anp.linalg.norm(F_u_k)
    if norm_F_u_k <= gamma * beta_k:
        mu_k = u_k
        beta_k = norm_F_u_k
        return mu_k, alpha_k, beta_k

    norm_d_k_sq = anp.linalg.norm(d_k)**2
    xi_k = - (F_u_k @ d_k) / norm_d_k_sq

    norm_F_u_k_sq = norm_F_u_k**2
    v_k = mu_k - ((F_u_k @ (mu_k - u_k)) / norm_F_u_k_sq) * F_u_k
    w_k = mu_k - delta * F_mu_k

    F_v_k = F(v_k, V_k, DI, basis_normal, B_k, gradf, lambda_reg, rho)
    norm_F_v_k = anp.linalg.norm(F_v_k)
    norm_F_mu_k = anp.linalg.norm(F_mu_k)

    if xi_k >= phi_1:
        if norm_F_v_k <= norm_F_mu_k:
            mu_k = v_k
        else:
            mu_k = w_k
    
    if xi_k >= phi_2:
        alpha_k = anp.random.uniform(bar_alpha, alpha_k)
    elif phi_1 <= xi_k < phi_2:
        alpha_k = anp.random.uniform(alpha_k, psi_1 * alpha_k)
    else:
        alpha_k = anp.random.uniform(psi_1 * alpha_k, psi_2 * alpha_k)

    return mu_k, alpha_k, beta_k

def compute_step_direction(J_k, F_mu_k, alpha_k, tau, I_s, how):
    """
    This function computes the descent direction at step k.

    INPUT
    =====
    - J_k: anp.array, shape (s, s). Evaluation of the Jacobian of F at mu_k.  
    - F_mu_k: anp.array, shape (s, s). Evaluation of F computed at mu_k. 
    - alpha_k: float>0. Value at iteration k.
    - tau: float>0. Threshold for verifying the inexactness criterion.
    - I_s: anp.array, shape (s, s). Identity matrix.
    - how: string. How to solve the system of linear equations for finding d_k. Choose "exactly" or "inexactly" via conjugate gradient.
    
    OUTPUT
    ======
    - d_k: anp.array, shape (s,). descent direction at step k. 
    """
    nu_k = alpha_k * anp.linalg.norm(F_mu_k)

    reg_J_k = J_k + nu_k * I_s
    
    rhs = -F_mu_k

    if how=="exactly":
        d_k = anp.linalg.solve(reg_J_k, rhs)
    else:
        d_k = anp.asarray(cg(reg_J_k, rhs)[0])
        
    r_k = reg_J_k @ d_k + F_mu_k
    norm_r_k = anp.linalg.norm(r_k)
    norm_d_k = anp.linalg.norm(d_k)
    threshold = tau * min(1, alpha_k * anp.linalg.norm(F_mu_k) * norm_d_k)

    if norm_r_k > threshold: print("Inexactness criterion not satisfied.")
        
    return d_k


def solve_subproblem(V_k, DI, basis_normal, B_k, gradf, mu_k, lambda_reg, rho, alpha_k, beta_k, I_s, tau, phi_1, phi_2, psi_1, psi_2, bar_alpha, gamma, delta, how):
    """
    This function solves the subproblem related to the gradient update.

    INPUT
    =====
    - V_k: anp.array, shape (l,h). Linear measurable map belonging to the Stiefel manifold at iteration k.  
    - DI: anp.array, shape (l,h). Penalty matrix. 
    - basis_normal: anp.array, shape (s, l, h). Basis for the normal space at V_k, where s=h(h+1)/2. 
    - B_k: anp.array, shape (s, l*h). Flattening of basis_normal obtained by vertically stacking the vectorization of the matrices constituting basis_normal at V_k. 
    - gradf: anp.array, shape (l, h). Gradient of the KL divergence at V_k.
    - mu_k: anp.array, shape (s,). Updated Lagrangian multiplier. 
    - lambda_reg: float>0. ell_1-norm penalty.  
    - rho: float>0. Augmented Lagrangian penalty.  
    - alpha_k: float>0. Value at iteration k. 
    - beta_k: float>0. Updated value for iteration k+1. 
    - I_s: anp.array, shape (s, s). Identity matrix., 
    - tau: float>0. Threshold for verifying the inexactness criterion. 
    - phi_1: float in (0, phi_2]. 
    - phi_2: float in (0, 1). 
    - psi_1: float in (1, psi_2). 
    - psi_2: float>1.
    - bar_alpha: float>0. 
    - gamma: float in (0,1). 
    - delta: float in (0, 1/omega). 
    - how: string. How to solve the system of linear equations for finding d_k. Choose "exactly" or "inexactly" via conjugate gradient.
    
    OUTPUT
    ======
    - G_k: anp.array, shape (s,s). Update for the gradient. 
    - mu_k: anp.array, shape (s,). Updated Lagrangian multiplier.
    - alpha_k: float>0. Updated value for iteration k+1. 
    - beta_k: float>0. Updated value for iteration k+1.
    """        
    F_mu_k = F(mu_k, V_k, DI, basis_normal, B_k, gradf, lambda_reg, rho)
    J_k = J(mu_k, V_k, DI, basis_normal, B_k, gradf, lambda_reg, rho)

    d_k = compute_step_direction(J_k, F_mu_k, alpha_k, tau, I_s, how)
    mu_k, alpha_k, beta_k = safeguard_step(V_k, DI, mu_k, d_k, F_mu_k, basis_normal, B_k, gradf, lambda_reg, rho, alpha_k, beta_k, phi_1, phi_2, psi_1, psi_2, bar_alpha, gamma, delta)
    
    B_mu = V_k - rho * (gradf - anp.tensordot(basis_normal, mu_k, axes=([0], [0])))
    P = soft_thresholding(B_mu, lambda_reg * rho *DI)
    G_k = P - V_k
    return G_k, mu_k, alpha_k, beta_k

def ls_condition(V_k, DI, covlow, covhigh, G_k, manifold, a, G_k_norm_squared, lambda_reg, rho):
    """
    This evaluates the progress for a candidate V_k and a certain scalar a for the Armijo linesearch procedure.

    INPUT
    =====
    - V_k: anp.array, shape (l,h). Linear measurable map belonging to the Stiefel manifold at iteration k.  
    - DI: anp.array, shape (l,h). Penalty matrix.  
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.
    - G_k: anp.array, shape (s,s). Update for the gradient.
    - manifold: pymanopt Riemannian submanifold object. 
    - a: float. Constant at which evaluate the progress condition.
    - G_k_norm_squared: float. Squared Frobenious norm of the updated gradient.
    - lambda_reg: float>0. ell_1-norm penalty.  
    - rho: float>0. Augmented Lagrangian penalty.

    OUTPUT
    ======
    - bool. Condition.
    """
    obj_V_k = objective(V_k, covlow, covhigh, DI, lambda_reg)
    V_k_candidate = manifold.retraction(V_k, a*G_k)
    obj_V_k_candidate = objective(V_k_candidate, covlow, covhigh, DI, lambda_reg)
    return obj_V_k_candidate > obj_V_k - a/(2*rho) * G_k_norm_squared 

def LinSEPAL_PG(covlow, covhigh, lambda_reg, D, how, L, gamma_line, tau_line, max_iter=1000, tol=1e-6, initialization='structural', V_init=None, seed=42, verbose=False):
    """
    This function implements the recursion for the ManPG (manifold projected gradient) method.

    INPUT
    =====
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.
    - lambda_reg: float>0. ell_1-norm penalty.  
    - D: anp.array, shape (l,h). Matrix providing the structural information used to build D^I= I - D. 
    - how: string. How to solve the system of linear equations for finding d_k. Choose "exactly" or "inexactly" via conjugate gradient. 
    - L: float. Lipschitz constant for the smooth term of the objective function. 
    - gamma_line: float>0. Constant for the Armijo stepsize procedure.
    - tau_line: float>0. Constant for adaptive stepsize.
    - max_iter: int. Maximum number of iterations to be performed. Default=1000.
    - tol: float. Stopping criterion f(V)<tol.
    - initialization: str. Initialization method for V. Can be 'structural', 'notstructural', 'provided).
    - V_init:anp.array, shape (l,h). Provided initialization for V. It only matters when 'initialization'=='provided.
    - seed: int. anp seed for the random module.
    - verbose: bool. If True additional information is printed during the optimization process.

    OUTPUT
    ======
    - V_k: anp.array, shape (l,h). Optimized linear measurable map belonging to the Stiefel manifold. 
    - iter: int. Number of iterations.
    - obj_val_series: anp.array, shape (max_iter+1,). Series of objective values.
    """

    l, h = D.shape
    assert l > h, "The dimension of the low-level SCM must be higher than the high-level one."

    anp.random.seed(seed)
    A = anp.random.uniform(-1.,1.,size=(l,h))
    
    if initialization=='structural':
        V_init = A*D 
        V_init /= anp.linalg.norm(V_init, axis=0)
    elif initialization=="notstructural":
        V_init, _ = anp.linalg.qr(A)
    elif initialization=="provided":
        assert V_init is not None, "If initialization='provided', V_init should be not None"
    else:
        print("initialization can be in ('strucutral', 'notstructural', 'provided')")
        return

    if verbose:
        print(V_init)
    
    DI = anp.ones_like(D) - D 
   
    V_k = V_init
    s = h*(h+1)//2
    manifold = Stiefel(l,h)
    I_s = anp.eye(s)
    rho = 1/L
    

    phi_1 = gamma = omega = tau = .5
    phi_2 = .9
    psi_2 = 2.
    psi_1 = uniform(1, psi_2)
    bar_alpha = 1.e-5
    delta = 1.

    mu_k = anp.zeros(s)
    basis_normal = compute_basis_normal_space(V_k)
    B_k = anp.vstack([b.flatten() for b in basis_normal])
    gradf = grad_f(V_k, covlow, covhigh)
    F_mu_k = F(mu_k, V_k, DI, basis_normal, B_k, gradf, lambda_reg, rho)
    alpha_k = 1.
    beta_k = anp.linalg.norm(F_mu_k)

    iter=0
    converged=False

    obj_val_series = anp.zeros(max_iter + 1)
    obj_val_series[0]+=f(V_k, covlow, covhigh)
    
    while iter<max_iter:
        G_k, mu_k, alpha_k, beta_k = solve_subproblem(V_k, DI, basis_normal, B_k, gradf, mu_k, lambda_reg, rho, alpha_k, beta_k, I_s, tau, phi_1, phi_2, psi_1, psi_2, bar_alpha, gamma, delta, how)
        G_k_norm_squared = anp.linalg.norm(G_k, ord='fro')**2
        a = 1
        linesearchflag = 0
        
        while ls_condition(V_k, DI, covlow, covhigh, G_k, manifold, a, G_k_norm_squared, lambda_reg, rho) and a>1.e-4: 
            a *= gamma_line
            linesearchflag = 1

        V_k = manifold.retraction(V_k, a*G_k)

        if linesearchflag==1 : 
            rho *= tau_line
        else: 
            rho = max(1/L, rho/tau_line) 

        if verbose:
            if (iter+1)%10==0: 
                print(G_k_norm_squared)  
                print(iter+1, a, rho)

        obj_val_series[iter+1]+=f(V_k, covlow, covhigh)
        
        if obj_val_series[iter+1] < tol: 
            converged=True
            print("Objective convergence at iteration {}: objective={}".format(iter+1, obj_val_series[iter+1]))
            break

        iter+=1
        
        basis_normal = compute_basis_normal_space(V_k)
        B_k = anp.vstack([b.flatten() for b in basis_normal])
        gradf = grad_f(V_k, covlow, covhigh)

    if not converged: print("Max number of iterations reached: objective {}".format(obj_val_series[iter]))

    return V_k, iter, obj_val_series