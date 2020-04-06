# CIFAR10
Analysis of models and implementation different algorithms with aim to improve state of art results on CIFAR-10 dataset

## Setup experiment environment
1. Latest versions of following libraries are used:
````
numpy
sklearn
tensorflow
matplotlib
````

2. Download the dataset to your CIFAR10 directory and add it to `.gitignore`.
    `````
    cd CIFAR10
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xf cifar-10-python.tar.gz
    rm cifar-10-python.tar.gz
    echo 'cifar-10-batches-py' > .gitignore
    `````

## REFERENCES
* Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
