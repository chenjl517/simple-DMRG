import numpy as np

def transformBase(U,A):
    return np.conj(np.transpose(U)) @ A @ U