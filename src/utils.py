import matplotlib.pyplot as plt
import CONSTANTS as c
from os import listdir
import re 
import pickle

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_history(hist):
    fig=plt.figure()
    ax=fig.add_subplot(121)
    bx=fig.add_subplot(122)
    ax.plot(hist.history['loss'],label='loss')
    ax.plot(hist.history['val_loss'],label='val_loss')
    ax.set_title('loss')
    ax.legend()
    bx.plot(hist.history['accuracy'],label='accuracy')
    bx.plot(hist.history['val_accuracy'],label='val_accuracy')
    bx.set_title('accuracy')
    bx.legend()

def latest_checkpoint(model_name, attempt):
    l = listdir(c.ROOT + 'checkpoints/')
    present = False
    e = []
    for f in l:
        if model_name+':a='+str(attempt)+'e=' in f:
            present = True
        else :
            continue
            
        crop = f[f.rfind('=') + 1 :f.rfind('.')]
        e.append(int(crop))

    if present:
        epoch = max(e)
        return epoch, checkpoint_name(model_name, attempt, epoch)

    else : 
        return 0, -1

def latest_record(model_name, attempt):
    rn = record_name(model_name, attempt)
    if rn in listdir(path = c.ROOT + 'logs/'):
        return rn
    else :
        return -1 

def checkpoint_name(model_name, attempt,epoch):
    return model_name + ':a=' + str(attempt) + 'e=' + str(epoch)

def record_name(model_name, attempt):
    return model_name+':a='+str(attempt)

def save_obj(obj, name ):
    print('Saving record...', end='')
    with open(c.ROOT + 'logs/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print('Done')

def load_obj(name ):
    with open(c.ROOT + 'logs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)