import numpy as np
from sklearn.datasets import make_spd_matrix as make_spd_matrix
import autograd.numpy as anp
from autograd.numpy.linalg import inv
import pandas as pd
import pickle

data_dir="data/"
    
def save_obj_parquet(obj, name, data_dir):
    """
    This function saves an object using parquet. (This is not sensitive to versioning.)

    INPUT
    =====
    obj: list/pd.dataframe/np.array. Object to be saved.
    name: str. Name for the file (without the extension).
    data_dir: str. Path where to place the file.
    """
    
    if isinstance(obj, list):
        df = pd.DataFrame(obj)
        df.to_parquet(data_dir + name + '.parquet')
    elif isinstance(obj, pd.DataFrame):
        obj.to_parquet(data_dir + name + '.parquet')
    elif isinstance(obj, np.ndarray):
        df = pd.DataFrame(obj)
        df.to_parquet(data_dir + name + '.parquet')
    else:
        raise ValueError("Unsupported data type")
    
def load_obj_parquet(name, data_dir):
    """
    This function loads an object using parquet.
    
    INPUT
    =====
    name: str. Name for the file (without the extension).
    data_dir: str. Path where to find the file.
    """
    try:
        df = pd.read_parquet(data_dir + name + '.parquet')
        if df.index.name is not None:
            df.reset_index(inplace=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError("Parquet file not found")

def masked_stiefel_matrix(B, seed, only_positive=False):
    """
    This function returns the groud-truth abstraction matrix V.

    INPUT
    =====
    - B: anp.array, shape (l,h). Support for the ground-truth abstraction matrix.
    - seed: int. anp seed for the random module.
    
    OUTPUT
    ======
    - V: anp.array, shape (l,h). Ground-truth abstraction matrix. 
    """
    
    l, h = B.shape
    assert l>h, "The dimension of the low-level SCM has to be higher than the high-level ones."
    assert anp.all(anp.logical_or(B == 0, B == 1)), "All entries of D must be either 0 or 1."

    anp.random.seed(seed)
    if only_positive: A = anp.random.rand(l, h)
    else: A = anp.random.uniform(-1.,1.,size=(l, h))
    
    V = A * B

    V/=anp.linalg.norm(V, axis=0)
    return V

def gen_covariances(V, seed):
    """
    This function generates the covariance matrices for the low-level and high-level SCMs.

    INPUT
    =====
    - V: anp.array, shape (l,h). Ground-truth abstraction matrix.
    - seed: int. anp seed for the random module.

    OUTPUT
    ======
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.
    """

    l, h = V.shape
    assert l>h, "The dimension of the low-level SCM has to be higher than the high-level ones."
    assert anp.allclose(V.T@V, anp.eye(h)), "V must belong to the Stiefel manifold."

    covlow = anp.asarray(make_spd_matrix(l, random_state=seed))
    covhigh = V.T @ covlow @ V
    
    assert anp.all(anp.linalg.eigvals(covhigh) > 0), "The covariance matrix for the high-level SCM is not positive definite."

    return covlow, covhigh

def stiefel_arc_length(A,B):
    """According to pg. 30 in 
    Edelman, Alan, Tomás A. Arias, and Steven T. Smith. "The geometry of algorithms with orthogonality constraints." SIAM journal on Matrix Analysis and Applications 20.2 (1998): 303-353.
    """
    _, S, _ = anp.linalg.svd(A.T@B)
    S = np.round(S, 3)
    theta = anp.nan_to_num(anp.arccos(S))
    return anp.linalg.norm(theta, ord=2)

def frobenious_abs_distance(V_pred, V_true, norm=True):
    if norm:
        return anp.linalg.norm(abs(V_true)- abs(V_pred))/anp.linalg.norm(abs(V_true))
    return anp.linalg.norm(abs(V_true)- abs(V_pred))

def commutation_matrix(l, h):
    
    v = anp.arange(l*h).reshape((l,h), order='F').T.ravel(order='F')
    
    return anp.eye(l*h)[v,:]

def matrix_R(l, h):
    R = anp.zeros((h, l*h))
    
    for i in range(h):
        R[i, i*l:(i+1)*l]+=1

    return R

def row_validity(A):
    row_sums = anp.sum(A != 0, axis=1)  
    valid_rows = anp.sum(row_sums == 1)  
    return valid_rows / A.shape[0]

def column_validity(A):
    column_sums = anp.sum(A != 0, axis=0)  
    valid_columns = anp.sum(column_sums >= 1)  
    return valid_columns / A.shape[1]

def constructiveness(A, alpha=0.5, beta=0.5):
    r_validity = row_validity(A)
    c_validity = column_validity(A)
    return alpha * r_validity + beta * c_validity

def f(V, covlow, covhigh):
    """This function computes the KL divergence between the high-level SCM probability measure and
    the projection of the low-level SCM probability measure onto R^h.

    INPUT
    =====
    - V: anp.array, shape (l,h). Linear measurable map belonging to the Stiefel manifold.
    - covlow: anp.array, shape (l,l). Covariance for the low-level SCM.
    - covhigh: anp.array, shape (h,h). Covariance for the high-level SCM.

    OUTPUT
    ======    
    - float. KL divergence. 
    """
    term1 = anp.trace(inv(V.T @ covlow @ V) @ covhigh)
    term2 = anp.log(anp.linalg.det(V.T @ covlow @ V))
    term3 = - anp.log(anp.linalg.det(covhigh))

    return term1 + term2 + term3 - V.shape[1]