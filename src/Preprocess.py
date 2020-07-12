import numpy as np
from sklearn.preprocessing import OneHotEncoder

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    return (x-min_val) / (max_val-min_val)

def normalize_elemental(x):
    # element wise normalize 
    raise('Not yet implemented!')

def ohe(y):
    return OneHotEncoder().fit_transform(np.array(y).reshape(-1,1)).toarray()

def todtype(x, dtype='float64'):
    return x.astype(dtype)


