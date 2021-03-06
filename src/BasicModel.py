import tensorflow as tf 
from CustomCallbacks import LongTermTrainer, SaveLosseAndMetrics
from utils import *

class FeatureBlock(tf.keras.Model):

    def __init__(self):
        super(FeatureBlock, self).__init__()
        
        # Block 1
        self.c11  = tf.keras.layers.Conv2D(35, (3,3))
        self.c12  = tf.keras.layers.Conv2D(40, (3,3))
        self.m11 = tf.keras.layers.MaxPool2D()
        self.c13  = tf.keras.layers.Conv2D(40, (1,1))
        
        # Block 2
        self.c21 = tf.keras.layers.Conv2D(40, (3,3))
        self.c22 = tf.keras.layers.Conv2D(40, (3,3))
        self.m21 = tf.keras.layers.MaxPool2D()
        self.c23  = tf.keras.layers.Conv2D(50, (1,1))

    def call(self, inputs):
        
        # Block 1
        x1 = self.c11(inputs)
        x1 = self.c12(x1)
        x1 = self.m11(x1)
        x1 = self.c13(x1)
        
        # Block 2
        x2 = tf.keras.layers.ZeroPadding2D(padding=1)(x1)
        x2 = self.c21(x2)
        x2 = tf.keras.layers.ZeroPadding2D(padding=1)(x2)
        x2 = self.c22(x2)     
        x2 = self.m21(x2)
        x2 = self.c23(x2)
        
        return x2

class DenseBlock(tf.keras.Model):

    def __init__(self):
        super(DenseBlock, self).__init__()
        
        self.f31 = tf.keras.layers.Flatten()
        self.d31 = tf.keras.layers.Dense(146, activation= tf.keras.activations.relu)
        self.d32 = tf.keras.layers.Dense(146, activation= tf.keras.activations.relu)
        self.d33 = tf.keras.layers.Dense(100, activation= tf.keras.activations.softmax)

    def call(self, inputs):
        
        x1 = self.f31(inputs)
        x1 = self.d31(x1)
        x1 = self.d32(x1)
        x1 = self.d33(x1)
        #x3 = tf.clip_by_value(x3, 1e-2, 1)
        
        return x1

class CompileModel(tf.keras.Model):

    def __init__(self):
        super(CompileModel, self).__init__()
        
        self.feature_block = FeatureBlock()
        self.dense_block = DenseBlock()
        self.model_name = 'Basic'

    def call(self, inputs):
        
        x1 = self.feature_block(inputs)
        x2 = self.dense_block(x1)
        
        return x2

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)

loss = tf.keras.losses.CategoricalCrossentropy()

class Attempt1(LongTermTrainer):
    
    def __init__(self):
        super(Attempt1, self).__init__()

        self.attempt = 1
        #self.model.optimizer.lr = 1e-2

    def lr_schedule(self, epoch, lr):
        return lr * time_decay(epoch)

attempts = {0: LongTermTrainer(), 1 : Attempt1(), }

# Add Callbacks below specially schedulers.
specific_callbacks = [SaveLosseAndMetrics(attempt=1)]

# Add preprocesses here, which will be done along with other preprocesses 
specific_preprocesses = ([], [])