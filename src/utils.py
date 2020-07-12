import matplotlib.pyplot as plt

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