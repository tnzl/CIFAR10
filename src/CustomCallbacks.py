import tensorflow as tf
import numpy as np
from utils import *
import CONSTANTS as c
from os import remove
# Use this as super class for specific callbacks to model
class LongTermTrainer(tf.keras.callbacks.Callback):

    def __init__(self):
        super(LongTermTrainer, self).__init__()
        self.epoch = None 
        self.attempt = 1
    
    # Add Schedules of different parameters here.
    
    def lr_schedule(self, epoch):
        #Schedule lr here in /child class
        return self.model.optimizer.lr

    def load_weights(self, attempt):
        e, lc = latest_checkpoint(self.model.model_name, self.attempt)
        self.epoch = e
        if lc != -1:
            print('Loading model...', end='')
            self.model.load_weights(c.ROOT + 'checkpoints/'+lc) 
            print('Done\nStarting from epoch = ',self.epoch)
        else:
            print('Starting to train new model')
    
    # Add pre-defined functions here
    
    def on_train_begin(self, logs=None):
        # create folder if not present
        self.load_weights(self.attempt)
        # load epochnumber and set learning rate and other related params
        pass 

    def on_epoch_begin(self, epoch, logs=None):
        # set learning rate 
        self.model.optimizer.lr = self.lr_schedule(self.epoch)
        
        # increment total epoch
        self.epoch += 1 

    def on_epoch_end(self, epoch, logs=None):
        # check conditions on epoch to stop training 
        print('\nSaving weights...', end='')
        self.model.save_weights(c.ROOT
                            + 'checkpoints/'
                            + checkpoint_name(self.model.model_name, self.attempt, self.epoch)) 
        print('Done')   
        
class SaveLosseAndMetrics(tf.keras.callbacks.Callback):

    def __init__(self, attempt=1, batch_interval=5):
        super(SaveLosseAndMetrics, self).__init__()
        self.batch_interval = batch_interval
        self.keys = ['loss']
        self.attempt = attempt
        self.record = None

    def on_train_begin(self, logs=None):
        
        lat_rec = latest_record(self.model.model_name, self.attempt)
        if lat_rec == -1:
            self.record = {}
        else :
            self. record = load_obj(c.ROOT + 'logs/' + lat_rec)
        
        self.keys += self.model.metrics_names
        
        for key in self.keys:
            if key not in self.record.keys():
                self.record[key] = []
            
    def on_train_batch_end(self, batch, logs=None):

        if batch % self.batch_interval:
            for key in self.keys:
                self.record[key].append(logs[key])
                
    def on_epoch_end(self, epoch, logs=None):

        # save record here.
        save_obj(self.record, record_name(self.model.model_name, self.attempt))
              
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