import DeepModel, BasicModel
from os import getcwd

IMG_SIZE = 32
CHANNELS = 3
ROOT = getcwd() + '/'

MODEL_MODULES = {'Deep' : DeepModel,'Basic' : BasicModel}
    