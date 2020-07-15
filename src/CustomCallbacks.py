import tensorflow as tf
import numpy as np
from utils import latest_checkpoint, checkpoint_name
import CONSTANTS as c

# Use this as super class for specific callbacks to model
class LongTermTrainer(tf.keras.callbacks.Callback):

    def __init__(self):
        super(LongTermTrainer, self).__init__()
        epoch = None 
    
    def lr_schedule(self, epoch):
        #Schedule lr here in /child class
        return self.model.optimizer.lr

    def on_train_begin(self, logs=None):
        # create folder if not present
        e, lc = latest_checkpoint(self.model.model_name)
        print("######",c.ROOT)
        self.epoch = e
        if lc != 'NO CKPT':
            print('Loading model...\nStarting from epoch ',self.epoch)

            self.model.load_weights(lc) 
        else:
            print('Starting to train new model')
        # load epochnumber and set learning rate and other related params

    def on_epoch_begin(self, epoch, logs=None):
        # set learning rate 
        self.model.optimizer.lr = self.lr_schedule(self.epoch)
        
        # increment total epoch
        self.epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        # check conditions on epoch to stop training 
        print('\nSaving weights...', end='')
        self.model.save_weights(checkpoint_name(self.model.model_name, 1, epoch)) 
        print('Done')   
        
class SaveLosseAndMetrics(tf.keras.callbacks.Callback):
    
    def __init__(self, batch_interval=5, load_record=None, save_record=None):
        super(SaveLosseAndMetrics, self).__init__()
        self.batch_interval = batch_interval
        self.record = {}
        self.keys = ['loss']
        self.save_record = save_record
        self.load_record = load_record
        
        if load_record :
            import pickle
            with open(self.load_record, 'rb') as f:
                 self.record = pickle.load(f)
            
    
    def on_train_begin(self, logs=None):
        self.keys += self.model.metrics_names
        
        for key in self.keys:
            if key not in self.record.keys():
                self.record[key] = []
            
    def on_train_batch_end(self, batch, logs=None):

        if batch % self.batch_interval:
            for key in self.keys:
                self.record[key].append(logs[key])
                
    def on_epoch_end(self, epoch, logs=None):
        if self.save_record:
            import pickle
            with open(self.save_record, 'wb') as f:
                    pickle.dump(self.save_record, f, pickle.HIGHEST_PROTOCOL)
                
class LRScheduler(tf.keras.callbacks.Callback):
    
    def __init__(self, fn):
        
        self.fn = fn
        self.schedule = []
        self.batches = None
    
    def on_epoch_begin(self, epoch, logs=None):
        #schedule here for lr of each batch 
        self.batches = something 
        b = list(range(batches))
        self.schedule = self.fn(b)
        
    def on_train_batch_begin(self, batch, logs=None):
        self.model.optimizer.lr = self.schedule[batch]
        # complete here
        
class SaveBestModel(tf.keras.callbacks.Callback):
    
    def __init__(self):
        super(SaveBestModel, self).__init__()
        
        self.paratha = []
    
    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        self.paratha.append(logs['loss'])