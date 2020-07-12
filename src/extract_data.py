    # Extract Matrices from datasets and save in inputs.
    # Create batches here if big dataset.
    # Do not apply any preprocesses here.

import os
import numpy as np
import CONSTANTS as c

if not os.path.exists(c.ROOT+'cifar10'):
    os.mkdir(c.ROOT+'cifar10')

if not os.path.exists(c.ROOT+'cifar100'):
    os.mkdir(c.ROOT+'cifar100')

if not os.path.exists(c.ROOT+'mnist-digit'):
    os.mkdir(c.ROOT+'mnistdigit')