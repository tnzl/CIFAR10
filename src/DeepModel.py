import tensorflow as tf

class FeatureBlock(tf.keras.Model):

    def __init__(self):
        super(FeatureBlock, self).__init__()
        
        # Block 1
        self.c11  = tf.keras.layers.Conv2D(96, 
                                           (3,3), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.c12  = tf.keras.layers.Conv2D(96, 
                                           (3,3), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.m11 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        self.dr11 = tf.keras.layers.Dropout(INIT_DROPOUT_RATE)
        
        # Block 2
        self.c21  = tf.keras.layers.Conv2D(96, 
                                           (1,1), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.c22  = tf.keras.layers.Conv2D(96, 
                                           (2,2), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.c23  = tf.keras.layers.Conv2D(96, 
                                           (2,2), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.m21 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        self.dr21 = tf.keras.layers.Dropout(INIT_DROPOUT_RATE)
        
        # Block 3
        self.c31  = tf.keras.layers.Conv2D(168, 
                                           (1,1), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.c32  = tf.keras.layers.Conv2D(168, 
                                           (2,2), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.c33  = tf.keras.layers.Conv2D(168, 
                                           (2,2), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.m31 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        self.dr31 = tf.keras.layers.Dropout(INIT_DROPOUT_RATE)
        
        # Block 4 
        self.c41  = tf.keras.layers.Conv2D(168, 
                                           (1,1), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.c42  = tf.keras.layers.Conv2D(168, 
                                           (2,2), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.c43  = tf.keras.layers.Conv2D(168, 
                                           (2,2), 
                                           padding='same', 
                                           kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY_RATE), 
                                           activation = tf.keras.activations.elu)
        self.m41 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        self.dr41 = tf.keras.layers.Dropout(INIT_DROPOUT_RATE)
        
    def call(self, inputs):
        
        # Block 1
        x1 = self.c11(inputs)
        x1 = self.c12(x1)
        x1 = self.m11(x1)
        x1 = self.dr11(x1)
        
        # Block 2
        x2 = self.c21(x1)
        x2 = self.c22(x2)
        x2 = self.c23(x2)
        x2 = self.m21(x2)
        x2 = self.dr21(x2)
        
        # Block 3
        x3 = self.c31(x2)
        x3 = self.c32(x3)
        x3 = self.c33(x3)
        x3 = self.m31(x3)
        x3 = self.dr31(x3)
        
        # Block 4
        x4 = self.c41(x3)
        x4 = self.c42(x4)
        x4 = self.c43(x4)
        x4 = self.m41(x4)
        x4 = self.dr41(x4)
        
        return x4

class DenseBlock(tf.keras.Model):

    def __init__(self):
        super(DenseBlock, self).__init__()
        
        self.f31 = tf.keras.layers.Flatten()
        self.d31 = tf.keras.layers.Dense(200, activation= tf.keras.activations.relu)
        self.d32 = tf.keras.layers.Dense(100, activation= tf.keras.activations.softmax)

    def call(self, inputs):
        
        x1 = self.f31(inputs)
        x1 = self.d31(x1)
        x1 = self.d32(x1)
        
        return x1

class CompileModel(tf.keras.Model):

    def __init__(self):
        super(BasicModel, self).__init__()
        
        self.feature_block = FeatureBlock2()
        self.dense_block = DenseBlock2()

    def call(self, inputs):
        
        x1 = self.feature_block(inputs)
        x2 = self.dense_block(x1)
        
        return x2


name = 'Deep'

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)

loss = tf.keras.losses.CategoricalCrossentropy()

# Add Callbacks below specially schedulers.
specific_callbacks = []

# Add preprocesses here, which will be done along with other preprocesses 
specific_preprocesses = ([], [])