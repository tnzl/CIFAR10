# CIFAR10
Analysis of models and implementation different algorithms with aim to improve state of art results on CIFAR-10 dataset

# Models
** Remove this line: Relatively light weight model to be chosen in order to apply **

Simple CNN: Enhanced Image Classification With a Fast-LearningShallow Convolutional Neural Network
https://arxiv.org/pdf/1503.04596.pdf

Resnet: Deep Residual Learning for Image Recognition https://arxiv.org/pdf/1512.03385.pdf

Read following if time.
Convolutional Kernel Networks
https://arxiv.org/pdf/1406.3332.pdf

# Training
    1. Batch Normalization 
    2. Transfer Learning: Deep Convolutional Neural Networks asGeneric Feature Extractors
    https://www.isip.uni-luebeck.de/fileadmin/uploads/tx_wapublications/hertel_ijcnn_2015.pdfa
    Pre-train on MNIST maybe or some else dataset.
    3.
  

# Overfitting
    1. Image augmentation 
    2. Regularization 
    3. Dropout
    4. Stochastic Pooling for Regularization ofDeep Convolutional Neural Networks https://arxiv.org/pdf/1301.3557.pdf 
    5. Fractional Max-Pooling: https://arxiv.org/pdf/1412.6071.pdf This because maxpooling gives spatial loss
    
# Hyperparameter tuning/ Optimization
    1. Practical Bayesian Optimization of MachineLearning Algorithms: http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
  

# Visualization
    1. Convolution layer visualization 
    2. pca or tSNE
    3. Clusterrning using knn. Create 3d graph using PCA to visualize it.
    4. Confusion matrix


# REFERENCES

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
* Blog: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html on visualizing convolution layers.
* Guide: http://karpathy.github.io/2019/04/25/recipe/ A Recipe for Training Neural Networks
