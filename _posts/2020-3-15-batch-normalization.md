---
title: Batch Normalization
tags: [Deep Learning, Artificial Intelligence, Batch]
style: fill
color: secondary
description: My Understanding about this Awesome Layer.
---
<!-- 
Source: [GitHub](https://github.com/amorehead/jazz-nn)

![](https://amorehead.github.io/assets/img/jazz_nn.jpg) -->

## Batch Normalization 

### Motivation:
Batch Normalization is basically one of the ideas that prevents gradient vanishing / gradient exploding. 
So far, this problem has been solved with changes in Activation function (ReLU, etc.), Careful Initialization, and small learning rate, 
but in this paper, it is a fundamental way to accelerate the learning speed by stabilizing the training process as a whole rather than these indirect methods.


The author of the paper <a href="https://arxiv.org/abs/1502.03167">Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift</a> 
argues that the reason for the destablization is the Internal Covariance Shift which caused by the variation of the distribution of input for each layer of the network and activation.
So to prevent this, we can think of a method of normalizing the distribution of the input of each layer to an input with an average of 0 and standard deviation of 1.
    
### Algorithm:

The outline of the algorithm is described as shown in the image below:
![image info](https://tuvovan.github.io/assets/img/bn_algorithm.png)

The Batch Normalization is applied before entering the specific hidden layers, it modifies the input and use the new values as the activation function.
When training with the training data, the mean and standard deviation are obtained from the current batch. However, for inference using the test data, instead of using 
the values of the mini-batch, the moving avergage is calculated and unbiased variance of the training data and do the same scale/shift to the testing data.


the process may be different when being applied to CNN. In general, the weight is applied in the form of Wx+b before adding the value to the normal activation function. 
If we want to use Batch Normalization, b must be removed. In addition, CNN wants to maintain the properties of convolution, it makes each batch normalization variables based on each channel.
For example, let's say the Batch Normalization is applied to a convolution layer with a mini-batch of m and a channel size of n, then there should be n different batch normalization variables.

## Benefits

In the current Deep Network, if the learning rate is too high, the gradient explode/vanishing or fall into localminima. When using Batch Normalization,
it is not affectd by the parameter scale when propagating, therefore the learning rate can be set greatly, which means faster training.

Batch Normalization has its own regulation properties which makes it possible to exclude Dropout or any other regulation terms.