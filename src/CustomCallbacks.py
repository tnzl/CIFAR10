import tensorflow as tf
import numpy as np

class LongTermTrainer(tf.keras.callbacks.Callback):

    def __init__(self, number of epochs to train this time, ):
        super(SaveLosseAndMetrics, self).__init__()
    def on_train_begin(self, something):
        create folder if not present
        load model 
        load epochnumber and set learning rate and other related params

    def on_epoch_begin(self, epoch):
    set learning rate 

    def on_epoch_end(smthng)
    save model
    check conditions on epoch to stop training 




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