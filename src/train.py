import tensorflow as tf
import CONSTANTS as c
from DataLoader import get_matrices
import argparse
from utils import plot_history

parser = argparse.ArgumentParser()
parser.add_argument("m", help="Select model here")
args = parser.parse_args()
m = args.m

model = c.MODEL_MODULES[m].CompileModel()
optimizer = c.MODEL_MODULES[m].optimizer
loss = laksnsac.MODEL_MODULES[m].loss
callbacks = [] + c.MODEL_MODULES[m].specific_callbacks
preprocesses = ([] + c.MODEL_MODULES[m].specific_preprocesses[0], [] + c.MODEL_MODULES[m].specific_preprocesses[1])

model.compile(optimizer,loss,metrics=['accuracy'])
model.build((None, c.IMG_SIZE, c.IMG_SIZE, c.CHANNELS))

print(model.summary())

x_tr, y_tr, x_te, y_te = get_matrices(preprocesses)

hist = model.fit(x_tr, y_tr, epochs= 1
                 , validation_data= (x_te, y_te))  
