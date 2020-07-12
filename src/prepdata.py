import numpy as np
import cv2
from sklearn.preprocessing import normalize

FUNCTION = type(lambda x: x)


class Data_processor():
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    IMG_SIZE = None
    # batches = 0
    preprocesses_x = None
    preprocesses_y = None
    minibatch_iterator = 0
    minibatch_size = 16
    minibatch_count = None
    doit = False

    def __init__(self, data_source, img_size, channels=3, batches=0, val_split=None, val_set=None, minibatch_size=16, minibatch_count=None, preprocesses_x=[], preprocesses_y=[], doit=False):
        self.doit = doit
        self.preprocesses_x = preprocesses_x
        self.preprocesses_y = preprocesses_y
        self.IMG_SIZE = img_size
        self.channels = channels
        self.train_x, self.train_y, self.val_x, self.val_y = self.split(
            data_source, batches, val_split, val_set)
        if self.doit:
            self.train_x, self.train_y = self.preprocess(
                (self.train_x, self.train_y), self.preprocesses_x, self.preprocesses_y)
        if self.val_x is not None:
            self.val_x, self.val_y = self.preprocess(
                (self.val_x, self.val_y), self.preprocesses_x, self.preprocesses_y)
        self.IMG_SIZE = img_size
        self.channels = channels
        self.minibatch_size = minibatch_size
        if minibatch_count != None:
            self.minibatch_size = len(self.train_y)//minibatch_count
            self.minibatch_count = minibatch_count
        else:
            self.minibatch_count = len(self.train_y)//self.minibatch_size

    def reset(self):
        self.minibatch_iterator = -1

    def split(self, data_source, batches, val_split, val_set):

        x = self.convert_to_array(data_source[0])
        y = self.convert_to_array(data_source[1])

        if batches != 0:
            xx = self.convert_to_array(x[0])
            yy = self.convert_to_array(y[0])
            print(111, yy.shape)
            for i in range(1, len(x)):
                xx = np.concatenate((xx, self.convert_to_array(x[i])))
                yy = np.concatenate((yy, self.convert_to_array(y[i])))
            x = xx
            y = yy

        if val_set != None:
            train_x = x
            train_y = y
            val_x = val_set[0]
            val_y = val_set[1]
        elif val_split != None:
            l = x.shape[0]
            train_x = x[:int((1-val_split)*l)]
            train_y = y[:int((1-val_split)*l)]
            val_x = x[int(val_split*l):]
            val_y = y[int(val_split*l):]
        else:
            train_x = x
            train_y = y
            val_x = None
            val_y = None

        return train_x, train_y, val_x, val_y

    def convert_to_array(self, x):
        if 'scipy.sparse' in str(type(x)):
            x = x.toarray()
            return x
        return np.array(list(x))

    def preprocess(self, data, preps_x, preps_y):
        all_preps_x = {
            'normalize': self.normalize,
            'add_border': self.add_border
        }
        all_preps_y = {
            'normalize': self.normalize,
            'one_hot_encode': self.one_hot_encode
        }
        x, y = data
        print('ggggg', x.shape)
        for prep in preps_x:
            fn = all_preps_x[prep]
            x = fn(x)
        for prep in preps_y:
            fn = all_preps_y[prep]
            x = fn(y)
        return x, y

    def get_train_batch(self):
        self.minibatch_iterator += 1
        if self.minibatch_iterator == self.minibatch_count:
            print("Epoch completed...")
            return None

        x, y = self.train_x[self.minibatch_iterator * self.minibatch_size: (
            self.minibatch_iterator + 1) * self.minibatch_size], self.train_y[self.minibatch_iterator * self.minibatch_size: (self.minibatch_iterator + 1) * self.minibatch_size]
        print('hhh', x.shape, y.shape)
        if not self.doit:
            x, y = self.preprocess(data=(x, y), preps_x=self.preprocesses_x,
                                   preps_y=self.preprocesses_y)
        return x, y

    def get_validattion_batch():
        pass

    def normalize(self, X):
        xx = []
        for img in X:
            xx.append((img-img.mean())/img.std())

        return np.array(xx)

    def batch_norm(X):
        pass

    def add_border(X):
        pass

    def one_hot_encode(Y):
        pass
