# module for fingerprint manipulation
import numpy as np

def sparse2dense(sparse):
    """
    Convert sparse fingerprint to dense fingerprint
    Args:
        sparse (np.array): sparse fingerprint
    Returns:
        dense (np.array): dense fingerprint
    """
    dense = []
    for idx, value in enumerate(sparse):
        if value == 1:
            dense.append(idx)
    return np.array(dense)
    
def dense2sparse(dense, fp_len=4860):
    """
    Convert dense fingerprint to sparse fingerprint
    Args:
        dense (np.array): dense fingerprint
        fp_len (int): length of the fingerprint
    Returns:
        sparse (np.array): sparse fingerprint
    """
    sparse = np.zeros(fp_len, dtype=np.int8)
    for value in dense:
        sparse[value] = 1
    return sparse