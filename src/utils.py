import matplotlib.pyplot as plt
import CONSTANTS as c
from os import listdir
import re 

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

def latest_checkpoint(model_name):
    # l = listdir(c.ROOT + 'checkpoints')
    # r = re.compile(model_name + "\d*")
    # newlist = list(filter(r.match, l))
    return 0, 'NO CKPT'

def checkpoint_name(model,attempt,epoch):
    return model+'a='+str(attempt)+'e='+str(epoch)