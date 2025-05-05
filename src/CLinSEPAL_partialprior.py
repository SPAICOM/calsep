import autograd.numpy as anp
from autograd.numpy.linalg import svd
from cvxpy import Minimize, Variable, quad_form, diag, Problem
from src.utils import commutation_matrix, f

def solve_QP(diagQ, c, A, G, v, w, solver="CLARABEL", verbose=False):
    """
    This function solves the QP subproblems. 
    """
    
    one = anp.ones_like(c)
    x = Variable(len(c), nonneg=True)
    objective = 0.5 * quad_form(x, diag(diagQ)) + c.T @ x
    pb = Problem(Minimize(objective), 
                 [A @ x == v,
                  - G @ x <= -w,
                  x <= one])
    
    try:
        pb.solve(solver, warm_start=False, verbose=False)
    except:
        if verbose: print("Failed OSQP")
        try:
            pb.solve(solver="CLARABEL", warm_start=False, verbose=False)
        except:
            if verbose: print("Failed OSQP")
            try:
                pb.solve(solver="OSQP", warm_start=False, verbose=False)
            except:
                if verbose: print("Failed OSQP")
                pb.solve(solver="CVXOPT", warm_start=False, verbose=False)
    return anp.array(x.value)

def gradient(A, B_prev, X_prev, covlow, covhigh):
    """
    This function computes the partial derivative of the objective function. This is used for both V and S, choosing values for B_prev and X_prev appropriately.
    """

    A0 = A*B_prev
    A1 = A0 * X_prev
    A2 = anp.linalg.inv(A1.T @ covlow @ A1)
    A3 = covlow @ A1 @ A2

    grad = 2*A3*A0 - 2*(A3 @ covhigh @ A2)*A0 
    return grad

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

def proximal_nonneg_hypersphere(X, M):
    """
    Compute the proximal operator for the unit hypersphere.

    INPUT
    =====
    - X: anp.array, shape (l,h). Splitting variable.
    - M: anp.array, shape (l,h). Structural prior matrix.
    
    OUTPUT
    ======
    - X_tilde: anp.array, shape (l,h). Proximal value.
    """
    _, l = X.shape
    X_tilde = anp.zeros_like(X)
    X_hat = anp.where(M.T!=0., X, -anp.inf)
    abs_dist = anp.abs(1.-X_hat)
    col_argmin = anp.argmin(abs_dist, axis=0)
    X_tilde[col_argmin, anp.arange(l)] = 1.
    return X_tilde

def CLinSEPAL_pp(covlow, covhigh, M, a, b, rho, adaptive_stepsize, tau=1.e-3, epsilon=.99, mu=10, tau_abs=1.e-4, tau_rel=1.e-4, max_iter=100, sca_iter=1000, sca_tol=1.e-3, seed=42, solver="CVXOPT", verbosity=2):
    """
    This function implements the CLinSEPAL recursion in case of partial prior structural knowledge of CA.

    INPUT
    =====
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.
    - M: anp.array, shape (l,h). Structural prior matrix.
    - a: float. Lower bound for CA coefficients. Useful to handle the product S*V.
    - b: float. Upper bound for CA coefficients. Useful to handle the product S*V.
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
    - solver: string. Solver for the QP problem. One among the cvxpy solvers.
    - verbosity: int. Level of information printed by the optimizer while it operates: 0 is silent, 2 is most verbose.

    OUTPUT
    ======
    - V: anp.array, shape (l,h). Optimized abstraction matrix.
    - S: anp.array, shape (l,h). Optimized support matrix.
    - Y1: anp.array, shape (l,h). Optimized splitting variable.
    - Y2: anp.array, shape (l,h). Optimized splitting variable.
    - X: anp.array, shape (l,h). Optimized splitting variable.
    - k+1: int. Number of iterations.
    - primal_res_seriesY1: anp.array, shape (max_iter+1,). Series of primal residuals realted to Y1.
    - primal_res_seriesY2: anp.array, shape (max_iter+1,). Series of primal residuals realted to Y2.
    - primal_res_seriesX: anp.array, shape (max_iter+1,). Series of primal residuals realted to X. 
    - dual_res_seriesY1: anp.array, shape (max_iter+1,). Series of dual residuals realted to Y1.
    - dual_res_seriesY2: anp.array, shape (max_iter+1,). Series of dual residuals realted to Y2.
    - dual_res_seriesX: anp.array, shape (max_iter+1,). Series of dual residuals realted to X.
    - obj_val_series: anp.array, shape (max_iter+1,). Series of objective values.
    """

    # Dimensions
    l, h = M.shape
    assert l > h, "The dimension of the low-level SCM must be higher than the high-level one."
    
    O = anp.ones_like(M)
    v = anp.ones(l)
    w = anp.ones(h)
    Klh = commutation_matrix(l,h)
    diag_vecM=anp.diag(M.flatten(order='F'))
    A = anp.kron(anp.ones(h).reshape(-1,1).T, anp.eye(l))@diag_vecM
    G = anp.kron(anp.ones(l).reshape(-1,1).T, anp.eye(h))@Klh@diag_vecM

    if verbosity>0: verbose=True
    else: verbose=False

    # Initialize variables
    anp.random.seed(seed)
    Y_init = anp.random.uniform(a,b,size=(l,h))
    Y1, _ = anp.linalg.qr(Y_init)
    Y2 = Y1.copy()
    
    S = M.copy()
    V = S*Y1
    X = (M*S).T

    scaledU1 = M*S*V - Y1
    scaledU2 = M*S*V - Y2
    scaledW = (M*S).T - X
    
    primal_res_seriesY1 = anp.zeros(max_iter + 1)
    primal_res_seriesY2 = anp.zeros(max_iter + 1)
    primal_res_seriesX = anp.zeros(max_iter + 1)
    dual_res_seriesY1 = anp.zeros(max_iter + 1)
    dual_res_seriesY2 = anp.zeros(max_iter + 1)
    dual_res_seriesX = anp.zeros(max_iter + 1)
    obj_val_series = anp.zeros(max_iter + 1)

    primal_res_seriesY1[0]+=anp.linalg.norm(M*S*V - Y1, 'fro')
    primal_res_seriesY2[0]+=anp.linalg.norm(M*S*V - Y2, 'fro')
    primal_res_seriesX[0]+=anp.linalg.norm((M*S).T - X, 'fro')
    dual_res_seriesY1[0]+=rho*anp.linalg.norm(M*S*Y1, ord='fro')
    dual_res_seriesY2[0]+=rho*anp.linalg.norm(M*V*Y2, ord='fro')
    dual_res_seriesX[0]+=anp.linalg.norm(X, ord='fro')
    obj_val_series[0]+=f(M*S*V, covlow, covhigh)

    if verbosity!=0:
        print("##### INITIALIZATION #####\n")
        print("Initial objective value: {}\n".format(obj_val_series[0]))
        print("Primal residual Y1 St(l,h): {}".format(primal_res_seriesY1[0]))
        print("Primal residual Y2 St(l,h): {}".format(primal_res_seriesY2[0]))
        print("Primal residual Sp(h,l): {}".format(primal_res_seriesX[0]))
        print("Dual residual Y1 St(l,h): {}".format(dual_res_seriesY1[0]))
        print("Dual residual Y2 St(l,h): {}".format(dual_res_seriesY2[0]))
        print("Dual residual Sp(l,h): {}".format(dual_res_seriesX[0]))

    converged=False

    for k in range(max_iter):
        S_prev = S.copy()
        Y1_prev = Y1.copy()
        Y2_prev = Y2.copy()
        X_prev = X.copy()

        V = update_V(V, covlow, covhigh, Y1, scaledU1, S, M, O, a, b, tau, rho, epsilon, sca_iter, sca_tol, verbosity)
        S = update_S(V, covlow, covhigh, Y2, scaledU2, S, X, scaledW, M, A, G, v, w, tau, rho, epsilon, sca_iter, sca_tol, solver, verbose)   
        Y1 = update_Y(V, scaledU1, S_prev, M)
        Y2 = update_Y(V, scaledU2, S, M)
        X = update_X(S, M, scaledW)

        scaledU1 += M*S_prev*V - Y1
        scaledU2 += M*S*V - Y2
        scaledW += (M*S).T - X

        MSprevV_norm = anp.linalg.norm(M*S_prev*V, ord='fro')
        MSV_norm = anp.linalg.norm(M*S*V, ord='fro')
        Y1_norm = anp.linalg.norm(Y1, ord='fro')
        Y2_norm = anp.linalg.norm(Y2, ord='fro')
        MS_norm = anp.linalg.norm((M*S).T, ord='fro')
        X_norm = anp.linalg.norm(X, ord='fro')
        MSprevU1_norm = anp.linalg.norm(M*S_prev*scaledU1, ord='fro')
        MVU2_norm = anp.linalg.norm(M*V*scaledU2, ord='fro')
        MW_norm = anp.linalg.norm(M.T*scaledW, ord='fro')

        primal_res_seriesY1[k+1] += anp.linalg.norm(M*S_prev*V - Y1, ord='fro')
        primal_res_seriesY2[k+1] += anp.linalg.norm(M*S*V - Y2, ord='fro')
        primal_res_seriesX[k+1] += anp.linalg.norm((M*S).T - X, ord='fro')
        dual_res_seriesY1[k+1] += rho*anp.linalg.norm(M*S_prev*(Y1-Y1_prev), ord='fro')
        dual_res_seriesY2[k+1] += rho*anp.linalg.norm(M*V*(Y2-Y2_prev), ord='fro')
        dual_res_seriesX[k+1] += rho*anp.linalg.norm(M*(X-X_prev).T, ord='fro')
        obj_val_series[k+1] += f(M*S*V, covlow, covhigh)

        primal_conditionY1 = primal_res_seriesY1[k+1] <= anp.sqrt(l*h)*tau_abs + tau_rel * rho* max(MSprevV_norm, Y1_norm)
        primal_conditionY2 = primal_res_seriesY2[k+1] <= anp.sqrt(l*h)*tau_abs + tau_rel * rho* max(MSV_norm, Y2_norm) 
        dual_conditionY1 = dual_res_seriesY1[k+1] <= anp.sqrt(l*h)*tau_abs + tau_rel * rho * MSprevU1_norm
        dual_conditionY2 = dual_res_seriesY2[k+1] <= anp.sqrt(l*h)*tau_abs + tau_rel * rho * MVU2_norm

        primal_conditionX = primal_res_seriesX[k+1] <= anp.sqrt(l*h)*tau_abs + tau_rel * max(MS_norm, X_norm) 
        dual_conditionX = dual_res_seriesX[k+1] <= anp.sqrt(l*h)*tau_abs + tau_rel * rho * MW_norm

        if primal_conditionY1 and primal_conditionY2 and primal_conditionX and dual_conditionY1 and dual_conditionY2 and dual_conditionX: 
            converged=True
            print("Residuals convergence at iteration {}: (objective, primal Y1 St, primal Y2 St, primal  X Sp, dual Y1 St, dual Y1 St, dual X Sp)=({},{},{},{},{},{},{})".format(k+1, obj_val_series[k+1], 
                                                                                                                                       primal_res_seriesY1[k+1],
                                                                                                                                       primal_res_seriesY2[k+1],
                                                                                                                                       primal_res_seriesX[k+1],
                                                                                                                                       dual_res_seriesY1[k+1],
                                                                                                                                       dual_res_seriesY2[k+1],
                                                                                                                                       dual_res_seriesX[k+1]))
            break

        if adaptive_stepsize:
            if max(primal_res_seriesY1[k+1],primal_res_seriesY2[k+1],primal_res_seriesX[k+1]) > mu*max(dual_res_seriesY1[k+1],dual_res_seriesY2[k+1],dual_res_seriesX[k+1]): 
                rho*=2
            elif max(dual_res_seriesY1[k+1],dual_res_seriesY2[k+1],dual_res_seriesX[k+1]) > mu*max(primal_res_seriesY1[k+1],primal_res_seriesY2[k+1],primal_res_seriesX[k+1]):
                rho/=2
            else:
                pass
        
        if verbosity!=0:
            if (k+1)%(max_iter//10)==0: 
                print("Iteration {}, objective value: {}".format(k+1, obj_val_series[k+1]))
                print("Primal residual Y1 St(l,h): {}".format(primal_res_seriesY1[k+1]))
                print("Primal residual Y2 St(l,h): {}".format(primal_res_seriesY2[k+1]))
                print("Primal residual X Sp(h,l): {}".format(primal_res_seriesX[k+1]))
                print("Dual residual Y1 St(l,h): {}".format(dual_res_seriesY1[k+1]))
                print("Dual residual Y2 St(l,h): {}".format(dual_res_seriesY2[k+1]))
                print("Dual residual X Sp(l,h): {}".format(dual_res_seriesX[k+1]))
                print("Rho: {}".format(rho))
    
    if not converged: print("Max number of iterations reached: (objective, primal Y1 St, primal Y2 St, primal  X Sp, dual Y1 St, dual Y1 St, dual X Sp)=({},{},{},{},{},{},{})".format(obj_val_series[k+1], 
                                                                                                                                                                                    primal_res_seriesY1[k+1],
                                                                                                                                                                                    primal_res_seriesY2[k+1],
                                                                                                                                                                                    primal_res_seriesX[k+1],
                                                                                                                                                                                    dual_res_seriesY1[k+1],
                                                                                                                                                                                    dual_res_seriesY2[k+1],
                                                                                                                                                                                    dual_res_seriesX[k+1]))
        
    return V, S, Y1, Y2, X, k+1, primal_res_seriesY1, primal_res_seriesY2, primal_res_seriesX, dual_res_seriesY1, dual_res_seriesY2, dual_res_seriesX, obj_val_series

def update_V(V, covlow, covhigh, Y1, scaledU1, S, M, O, a, b, tau, rho, epsilon, sca_iter, sca_tol, verbosity):
    """
    This function solves the update for V.

    INPUT
    =====
    - V: anp.array, shape (l,h). V matrix at previous iteration.
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.
    - Y1: anp.array, shape (l,h). Splitting variable.
    - scaledU1: anp.array, shape (l,h). Scaled dual variable associated with Y1.
    - S: anp.array, shape (l,h). S matrix at previous iteration.
    - M: anp.array, shape (l,h). Structural prior matrix.
    - O: anp.array, shape (l,h). Matrix of 1s.
    - a: float. Lower bound for CA coefficients. Useful to handle the product S*V.
    - b: float. Upper bound for CA coefficients. Useful to handle the product S*V.
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
        
        grad_k = gradient(M, S, V_prev, covlow, covhigh) 
        
        den = rho*M*S*S + tau*O
        num = rho*M*S*Y1 - rho*M*S*scaledU1 + tau*V_prev - grad_k
        V = anp.divide(num,den)
        V = anp.where(V>b,b,anp.where(V<a,a,V))
        V = V_prev + gamma *(V - V_prev)

        if anp.linalg.norm(V-V_prev,'fro')<sca_tol: 
            if verbosity>1:
                print("Subproblem V converged at iteration {}".format(counter+1))
            return V
        
        gamma *= (1- epsilon*gamma)
    
    if verbosity>1: print("Subproblem V didn't converge")
    return V 

def update_S(V, covlow, covhigh, Y2, scaledU2, S, X, scaledW, M, A, G, v, w, tau, rho, epsilon, sca_iter, sca_tol, solver, verbose):
    """
    This function solves the update for V.
    """
    l,h = V.shape
    gamma = (1-.01)*1./epsilon
    vecM = M.flatten(order='F')
    vecV = V.flatten(order='F')
    vecMV = vecM*vecV
    vecYU = (Y2 - scaledU2).flatten(order='F') 
    vecXW = (X - scaledW).T.flatten(order="F")
    vecYU_vecMV = vecYU * vecMV
    one = anp.ones(l*h)

    for counter in range(sca_iter):  
        S_prev = S.copy()
        grad_k = gradient(M, V, S_prev, covlow, covhigh) 
        
        vecS_prev = S_prev.flatten(order='F')
        vecgrad_k = grad_k.flatten(order='F')

        diagQ = tau * one + rho*vecMV*vecMV + rho *vecM*vecM
        c = vecgrad_k - tau*vecS_prev - rho*vecYU_vecMV - rho* vecM* vecXW

        ##### solve the costrained quadratic problem #####
        vecS = solve_QP(diagQ, c, A, G, v, w, solver, verbose)
        ##################################################

        S = S_prev + gamma * (vecS.reshape((l,h), order='F') - S_prev)

        dist_S=anp.linalg.norm(S-S_prev,'fro')

        if dist_S<sca_tol: 
            if verbose:
                print("Subproblem S convergence at iteration {}".format(counter+1))
            return S
        
        gamma *= (1- epsilon*gamma)
    if verbose>1: print("Subproblem V didn't converge")   
    return S 

def update_Y(V, scaledU, S, M):
    """
    This function solves the update for Ys. This works for both Y1 and Y2.

    INPUT
    =====
    - V: anp.array, shape (l,h). Abstraction matrix.
    - scaledU: anp.array, shape (l,h). Scaled dual variable.
    - S: anp.array, shape (l,h). Support of V^*.
    - M: anp.array, shape (l,h). Structural prior.

    OUTPUT
    ======
    - anp.array, shape (l,h). Update for the splitting variables Y.
    """

    return proximal_stiefel(M*S*V+scaledU)

def update_X(S, M, scaledW):
    """
    This function solves the update for X.

    INPUT
    =====
    - S: anp.array, shape (l,h). Current value for the support matrix S.
    - M: anp.array, shape (l,h). Structural prior matrix. 
    - scaledW: shape (h,l). Scaled dual variables associated with X. 

    OUTPUT
    ======
    - anp.array, shape (h,l). Updated X matrix.
    """
    return proximal_nonneg_hypersphere((M*S).T+scaledW, M)