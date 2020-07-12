# import sys
# sys.path.append('/home/mobiliser/Projects/Python Scripts/DL/CIFAR10/')
# import CIFAR10

#from CIFAR10.load_data import get_matrices

from  BasicModel import BasicModel
import tensorflow as tf
import numpy as np

tf.random.set_seed(1234)

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, TensorBoard, ModelCheckpoint

IMG_SIZE = 32
LR = 3e-3
CHANNELS = 3

# def preprocess(x, y):
    
#     # ohe
#     y = OneHotEncoder().fit_transform(np.array(y).reshape(-1,1)).toarray()
        
#     # as float16
#     x = x.astype('float64')
#     y = y.astype('float64')
    
#     # normalize
#     min_val = np.min(x)
#     max_val = np.max(x)
#     x = (x-min_val) / (max_val-min_val)
    
#     return x, y       

# def get_matrices():
    
#     (x_tr, y_tr), (x_te, y_te) = load_batch(999, test=True)
#     x_te, y_te = preprocess(x_te, y_te)
#     x_tr, y_tr = preprocess(x_tr, y_tr)
    
#     print('x_tr shape: '+ str(x_tr.shape))
#     print('y_tr shape: '+ str(y_tr.shape))
#     print('\nx_tr dtype: '+ str(x_tr.dtype))
#     print('y_tr dtype: '+ str(y_tr.dtype))
#     print('\nx_te shape: '+ str(x_te.shape))
#     print('y_te shape: '+ str(y_te.shape))
#     print('\nx_te dtype: '+ str(x_te.dtype))
#     print('y_te dtype: '+ str(y_te.dtype))  
    
#     return x_tr, y_tr, x_te, y_te
    
# # Load dataset from memory
# def load_batch(i, test = False):
#     return tf.keras.datasets.cifar100.load_data()


if __name__ == "__main__":

    print('Here: 2')

    x_tr, y_tr, x_te, y_te = get_matrices()

    model = BasicModel()
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3)
    loss=tf.keras.losses.CategoricalCrossentropy()
    #cb1 = SaveLosseAndMetrics()
    model.compile(optimizer,loss,metrics=['accuracy'])
    # model.build((None, IMG_SIZE, IMG_SIZE, CHANNELS))
    # model.summary()

    hist = model.fit(x_tr, y_tr, epochs= 1
                    , validation_data= (x_te, y_te))  

    print("--------Training complete--------")