import autograd.numpy as anp
from autograd.numpy.linalg import svd
from src.utils import f

def proximal_stiefel(X):
    """
    Compute the proximal operator for the Stiefel manifold.

    INPUT
    =====
    - X: anp.array. Argument.

    OUTPUT
    ======
    - X: anp.array. Proximal onto the Stiefel matrix.
    """
    U, _, Vt = svd(X, full_matrices=False)
    X = U @ Vt 
    return X

def CLinSEPAL_fp(covlow, covhigh, S, rho, adaptive_stepsize, tau=1.e-3, epsilon=.99, mu=10, tau_abs=1.e-4, tau_rel=1.e-4, max_iter=100, sca_iter=1000, sca_tol=1.e-3, seed=42, verbosity=2):
    """
    This function implements the CLinSEPAL recursion in case of full prior structural knowledge of CA.

    INPUT
    =====
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.
    - S: anp.array, shape (l,h). Structural prior matrix.
    - rho: float>0. Augmented Lagrangian penalty.
    - adaptive_stepsize: bool. If True uses ADMM adaptive stepsize strategy. Be careful, there are no convergence guarantees for ADMM with adaptive stepsize (Suggested 'False'). 
    - tau: float. Penalty parameter for SCA strongly-convex surogate.
    - epsilon: float. Epsilon parameter for the diminishing stepsize rule.
    - mu: float. mu parameter for adaptive stepsize strategy.
    - tau_abs: float. Absolute tolerance for primal and dual residuals convergence. 
    - tau_rel: float. Relative tolerance for primal and dual residuals convergence.
    - max_iter: int. Maximum number of iterations.
    - sca_iter: int. Max number of SCA iterations.
    - sca_tol: float. Tolerance for SCA convergence.
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
    l, h = S.shape
    assert l > h, "The dimension of the low-level SCM must be higher than the high-level one."
    O = anp.ones_like(S)

    # Initialize variables
    anp.random.seed(seed)
    A = anp.random.uniform(-1.,1.,size=(l,h))
    
    Y = A*S 
    Y/=anp.linalg.norm(Y, axis=0)
    
    V = Y.copy()

    scaledU = S*V - Y
    
    primal_res_series = anp.zeros(max_iter + 1)
    dual_res_series = anp.zeros(max_iter + 1)
    obj_val_series = anp.zeros(max_iter + 1)

    primal_res_series[0]+=anp.linalg.norm(S*V - Y, 'fro')
    dual_res_series[0]+= rho * anp.linalg.norm(S*Y, ord='fro')
    obj_val_series[0]+=f(S*V, covlow, covhigh)

    if verbosity!=0:
        print("##### INITIALIZATION #####\n")
        print("Initial objective value: {}\n".format(obj_val_series[0]))
        print("Primal residual: {}".format(primal_res_series[0]))
        print("Dual residual: {}".format(dual_res_series[0]))

    converged=False

    for k in range(max_iter):
        Y_prev = Y.copy()

        V = update_V(V, covlow, covhigh, Y, scaledU, S, O, tau, rho, epsilon, sca_iter, sca_tol, verbosity)
        
        Y = update_Y(V, scaledU, S)

        scaledU += S*V - Y

        SV_norm = anp.linalg.norm(S*V, ord='fro')
        Y_norm = anp.linalg.norm(Y, ord='fro')
        SU_norm = anp.linalg.norm(S*scaledU, ord='fro')

        primal_res_series[k+1] += anp.linalg.norm(S*V - Y, ord='fro')
        dual_res_series[k+1] += rho*anp.linalg.norm(S*(Y-Y_prev), ord='fro')
        obj_val_series[k+1] += f(S*V, covlow, covhigh)

        primal_condition = primal_res_series[k+1] <= anp.sqrt(l*h)*tau_abs + tau_rel * max(SV_norm, Y_norm) 
        dual_condition = dual_res_series[k+1] <= anp.sqrt(l*h)*tau_abs + tau_rel * rho * SU_norm

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

def update_V(V, covlow, covhigh, Y, scaledU, S, O, tau, rho, epsilon, sca_iter, sca_tol, verbosity):
    """
    This function solves the update for V.

    INPUT
    =====
    - V: anp.array, shape (l,h). V matrix at previous iteration.
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.
    - Y: anp.array, shape (l,h). Splitting variable.
    - scaledU: anp.array, shape (l,h). Scaled dual variable.
    - S: anp.array, shape (l,h). S matrix at previous iteration.
    - O: anp.array, shape (l,h). Matrix of 1s.
    - tau: float. Penalty parameter for SCA strongly-convex surogate.
    - rho: float. Augmented Lagrangian penalty.
    - epsilon: float. Epsilon parameter for the diminishing stepsize rule.
    - sca_iter: int. Max number of SCA iterations.
    - sca_tol: float. Tolerance for SCA convergence.
    - verbosity: Level of information printed by the optimizer while it operates: 0 is silent, 2 is most verbose.

    OUTPUT
    ====== 
    - V: anp.array, shape (l,h). Updated V matrix. 
    """

    gamma = (1-.01)*1./epsilon
    
    for counter in range(sca_iter):  
        V_prev = V.copy()
        
        A0 = S * V_prev
        A1 = anp.linalg.inv(A0.T @ covlow @ A0)
        A2 = covlow @ A0 @ A1

        grad_k = 2*A2*S - 2*(A2 @ covhigh @ A1)*S 
        alpha=1.
        den = rho*S*S + tau*O
        num = rho*S*Y - rho*S*scaledU + tau*V_prev - alpha*grad_k
        V = anp.divide(num,den)

        V = V_prev + gamma *(V - V_prev)

        if anp.linalg.norm(V-V_prev,'fro')<sca_tol: 
            if verbosity>1:
                print("Subproblem convergence at iteration {}".format(counter+1))
            return V
        
        gamma *= (1- epsilon*gamma)
        
    return V 

def update_Y(V, scaledU, S):
    """
    This function solves the update for Y.

    INPUT
    =====
    - V: anp.array, shape (l,h). Updated abstraction matrix.
    - scaledU: anp.array, shape (l,h). Scaled dual variable.
    - S: anp.array, shape (l,h). Support of V^*.

    OUTPUT
    ======
    - anp.array, shape (l,h). Update for the splitting variable Y.
    """

    return proximal_stiefel(S*V+scaledU)