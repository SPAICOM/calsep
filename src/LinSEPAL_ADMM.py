import pymanopt
from pymanopt.optimizers import ConjugateGradient
from pymanopt.manifolds import Stiefel
from pyproximal import L1 
import autograd.numpy as anp
from autograd.numpy.linalg import inv
from src.utils import f

def LinSEPAL_ADMM(covlow, covhigh, D, lambda_reg, rho, initialization, adaptive_stepsize, mu=10, tau_abs=1.e-4, tau_rel=1.e-4, max_iter=100, seed=42, verbosity=2):
    """
    This function implements the LinSEPAL-ADMM recursion.

    INPUT
    =====
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.
    - D: anp.array, shape (l,h). Matrix providing the structural information used to build D^I= I - D.
    - lambda_reg: float>0. ell_1-norm penalty.
    - rho: float>0. Augmented Lagrangian penalty.
    - initialization: str. Either 'structural' or 'notstructural'.
    - adaptive_stepsize: bool. If True uses ADMM adaptive stepsize strategy. Be careful, there are no convergence guarantees for ADMM with adaptive stepsize (Suggested 'False'). 
    - mu: float. mu parameter for adaptive stepsize strategy.
    - tau_abs: float. Absolute tolerance for primal and dual residuals convergence. 
    - tau_rel: float. Relative tolerance for primal and dual residuals convergence.
    - max_iter: int. Maximum number of iterations.
    - seed: int. anp seed for the random module.
    - verbosity: int. Level of information printed by the optimizer while it operates: 0 is silent, 2 is most verbose.

    OUTPUT
    ======
    - V: anp.array, shape (l,h). Optimized abstraction matrix.
    - Y: anp.array, shape (l,h). Optimized splitting variable.
    - k+1: int. Number of iterations.
    - primal_res_series: anp.array, shape (max_iter+1,). Series of primal residuals. 
    - dual_res_series: anp.array, shape (max_iter+1,). Series of dual residuals. 
    - obj_val_series: anp.array, shape (max_iter+1,). Series of objective values.
    """
    # Dimensions
    l, h = D.shape
    assert l > h, "The dimension of the low-level SCM must be higher than the high-level one."

    # Initialize variables
    anp.random.seed(seed)
    A = anp.random.uniform(-1.,1.,size=(l,h))
    
    if initialization=='structural':
        V = A*D 
        V/=anp.linalg.norm(V, axis=0)
    else:
        V, _ = anp.linalg.qr(A)

    if verbosity==2:
        print(V)
    
    DI = anp.ones_like(D) - D     
    Y = DI*V
    scaledU = anp.zeros((l, h))
    
    # Stiefel manifold for V
    manifold = Stiefel(l, h)

    primal_res_series = anp.zeros(max_iter + 1)
    dual_res_series = anp.zeros(max_iter + 1)
    obj_val_series = anp.zeros(max_iter + 1)

    primal_res_series[0]+=anp.linalg.norm(DI*V - Y, 'fro')
    dual_res_series[0]+=rho*anp.linalg.norm(DI*Y, ord='fro')
    obj_val_series[0]+=f(V, covlow, covhigh)

    if verbosity!=0:
        print("##### INITIALIZATION #####\n")
        print("Initial objective value: {}\n".format(obj_val_series[0]))
        print("Primal residual: {}".format(primal_res_series[0]))
        print("Dual residual: {}".format(dual_res_series[0]))

    converged=False

    # MADMM iterations
    for k in range(max_iter):

        Y_prev = Y.copy()
        
        # Update V using Riemannian gradient descent on the Stiefel manifold
        V = update_V(covlow, covhigh, Y, scaledU, DI, rho, manifold, verbosity)
        
        # Update Y using soft-thresholding
        Y = update_Y(V, scaledU, lambda_reg, rho, DI)

        # Update dual variable scaledU
        scaledU += DI*V - Y

        DIV_norm = anp.linalg.norm(DI*V, ord='fro')
        Y_norm = anp.linalg.norm(Y, ord='fro')
        DIU_norm = anp.linalg.norm(DI*scaledU, ord='fro')

        primal_res_series[k+1] += anp.linalg.norm(DI*V - Y, ord='fro')
        dual_res_series[k+1] += rho*anp.linalg.norm(DI*(Y-Y_prev), ord='fro')
        obj_val_series[k+1] += f(V, covlow, covhigh)
        
        primal_condition = primal_res_series[k+1] <= anp.sqrt(l*h)*tau_abs + tau_rel * max(DIV_norm, Y_norm) 
        dual_condition = dual_res_series[k+1] <= anp.sqrt(l*h)*tau_abs + tau_rel * rho * DIU_norm

        if primal_condition and dual_condition: 
            converged=True
            print("Residuals convergence at iteration {}: (objective, primal, dual)=({},{},{})".format(k+1, obj_val_series[k+1], primal_res_series[k+1], dual_res_series[k+1]))
            break
        
        if adaptive_stepsize:
            if primal_res_series[k+1] > mu*dual_res_series[k+1]: 
                rho*=2
            elif dual_res_series[k+1] > mu*primal_res_series[k+1]:
                rho/=2
            else:
                pass
        
        if verbosity!=0:
            if (k+1)%(max_iter//10)==0: 
                print("Iteration {}, objective value: {}".format(k+1, obj_val_series[k+1]))
                print("Primal residual: {}".format(primal_res_series[k+1]))
                print("Dual residual: {}".format(dual_res_series[k+1]))
                print("Rho: {}".format(rho))
    
    if not converged: print("Max number of iterations reached: (objective, primal, dual)=({},{},{})".format(obj_val_series[k+1], primal_res_series[k+1], dual_res_series[k+1]))

    return V, Y, k+1, primal_res_series, dual_res_series, obj_val_series

def update_V(covlow, covhigh, Y, scaledU, DI, rho, manifold, verbosity):
    """
    This function solves the update for V.

    INPUT
    =====
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.
    - Y: anp.array, shape (l,h). Splitting variable.
    - scaledU: anp.array, shape (l,h). Scaled dual variable.
    - DI: anp.array, shape (l,h). Penalty matrix.
    - rho: float. Augmented Lagrangian penalty.
    - manifold: pymanopt Riemannian submanifold object.
    - verbosity: Level of information printed by the optimizer while it operates: 0 is silent, 2 is most verbose.

    OUTPUT
    ======
    - anp.array, shape (l,h). Update for the abstraction matrix V.
    """

    # decorated obj. fun. for the update
    @pymanopt.function.autograd(manifold)
    def objective(V):
        term1 = anp.trace(inv(V.T @ covlow @ V) @ covhigh)
        term2 = anp.log(anp.linalg.det(V.T @ covlow @ V))
        term3 = 0.5 * rho * anp.linalg.norm(DI * V - Y + scaledU, 'fro')**2
        return term1 + term2 + term3
    
    # decorated gradient function 
    # corresponding to the update
    @pymanopt.function.autograd(manifold)
    def gradient(V):
        _, h = V.shape
        grad = 2 * covlow @ V @ inv(V.T @ covlow @ V) @ (-covhigh @ inv(V.T @ covlow @ V) + anp.eye(h))  + rho * (DI * V - Y + scaledU) * DI
        return grad
    
    # Use conjugate gradient to solve for V on the Stiefel manifold
    problem = pymanopt.Problem(manifold=manifold, cost=objective, euclidean_gradient=gradient)
    solver = ConjugateGradient(verbosity=verbosity)
        
    return solver.run(problem).point

def update_Y(V, scaledU, lambda_reg, rho, DI):
    """
    This function solves the update for Y using the element-wise soft-thresholding operator.

    INPUT
    =====
    - V: anp.array, shape (l,h). Updated abstraction matrix.
    - scaledU: anp.array, shape (l,h). Scaled dual variable.
    - lambda_reg: float>0. ell_1-norm penalty.
    - rho: float>0. Augmented Lagrangian penalty.
    - DI: anp.array, shape (l,h). Penalty matrix.

    OUTPUT
    ======
    - anp.array, shape (l,h). Update for the splitting variable Y.
    """
    
    return L1(sigma=lambda_reg).prox(DI*V + scaledU, 1./rho)
