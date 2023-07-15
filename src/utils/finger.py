# module for fingerprint manipulation
import numpy as np

def sparse2dense(sparse):
    dense = []
    for idx, value in enumerate(sparse):
        if value == 1:
            dense.append(idx)
    return np.array(dense)
    
def dense2sparse(dense, fp_len=4860):
    sparse = np.zeros(fp_len, dtype=np.int8)
    for value in dense:
        sparse[value] = 1
    return sparse