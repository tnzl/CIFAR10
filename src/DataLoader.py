from sklearn.preprocessing import OneHotEncoder
import numpy as np
from Preprocess import ohe, normalize, todtype
def preprocess(x, y, preprocesses):
    
    for process in preprocesses[0]:
        x = process(x)

    for process in preprocesses[1]:
        y = process(y)
    
    return x, y 

def get_matrices(preprocesses):
    
    (x_tr, y_tr), (x_te, y_te) = load_batch(test=True)
    
    preprocesses = ([todtype, normalize] + preprocesses[0], [ohe] + preprocesses[1])
    
    x_te, y_te = preprocess(x_te, y_te, preprocesses)
    x_tr, y_tr = preprocess(x_tr, y_tr, preprocesses)
    
    print('x_tr shape: '+ str(x_tr.shape))
    print('y_tr shape: '+ str(y_tr.shape))
    print('\nx_tr dtype: '+ str(x_tr.dtype))
    print('y_tr dtype: '+ str(y_tr.dtype))
    print('\nx_te shape: '+ str(x_te.shape))
    print('y_te shape: '+ str(y_te.shape))
    print('\nx_te dtype: '+ str(x_te.dtype))
    print('y_te dtype: '+ str(y_te.dtype)) 

    return x_tr, y_tr, x_te, y_te


# Load dataset from memory
def load_batch(test = False):
    from tensorflow.keras.datasets import cifar100
    return cifar100.load_data()