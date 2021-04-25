import numpy as np

def normalize_by_std(data_in):
    """
    std normalization
    """
    data_in -= np.mean(data_in, axis=0 ,keepdims=True)
    t_std = np.std(data_in, axis = 0, keepdims=True)
    t_std[t_std == 0] = 1.0
    data_in /= t_std

    return data_in
