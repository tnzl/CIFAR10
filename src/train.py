import tensorflow as tf
import CONSTANTS as c
from DataLoader import get_matrices
import argparse
from utils import plot_history

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Select model here")
parser.add_argument('-attempts', type=int, default=0, help='Set attempt')
parser.add_argument('-epochs', default=2, help='Set epochs')
args = parser.parse_args()
m = args.model
a = args.attempts
e = int(args.epochs)

print(m,a,e)

module = c.MODEL_MODULES[m]
model = module.CompileModel()
optimizer = module.optimizer
loss = module.loss
callbacks = [module.attempts[a]] + module.specific_callbacks
preprocesses = ([] + c.MODEL_MODULES[m].specific_preprocesses[0], [] + c.MODEL_MODULES[m].specific_preprocesses[1])

model.compile(optimizer, loss, metrics=['accuracy'])
model.build((None, c.IMG_SIZE, c.IMG_SIZE, c.CHANNELS))

print(model.summary())
x_tr, y_tr, x_te, y_te = get_matrices(preprocesses)

hist = model.fit(x_tr, y_tr, epochs= e
                 , validation_data= (x_te, y_te)
                 , callbacks = callbacks)  
