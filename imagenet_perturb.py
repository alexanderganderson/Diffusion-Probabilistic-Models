"""
We can perturb the diffusion kernel by an arbitrary function r(x), so that we
now sample from p^tilde(x) = (1 / Z) r(x) p(x). Here, we prepare such a
function where r(x) is a classifier of the form r(x) = p(y_0|x), and we mix
in the perturbation pas r(x^t) = r(x) ** (T-t)/T

Convolutional network from sklearn-theano
"""
import theano
import logging
from argparse import ArgumentParser

import numpy as np
import os

from theano import tensor
import theano.tensor as T

from sklearn_theano.feature_extraction.caffe.googlenet import create_theano_expressions


def build_classifier_grad(dataset, label=2, insert_sigmoid=False):
    """
    Loads a classifier, and builds functions p(y_label|x) and
        d log(p(y_label|x))/dx where x is the image

    ----------
    Parameters
    ----------
    dataset : string
         Dataset name, this function only works with IMAGENET
    label : int
         Integer determining which class
    insert_sigmoid : bool
        Flag for replacing Softmax with Sigmoid.
    """
    if dataset != 'IMAGENET':
        raise ValueError('This perturbation only works with IMAGENET')
    blobs, data_inputs = create_theano_expressions()
    # blobs are: b,c,0,1
    if insert_sigmoid:
        y_hat = T.nnet.sigmoid(blobs['loss3/classifier'].mean(axis=(2, 3)))
    else:
        y_hat = blobs['loss3/loss3'].mean(axis=(2, 3))
    pk_grad = T.grad(T.sum(T.log(y_hat[:, label])), data_inputs)  # Trick for dy[i]/dx[i]

    pk_grad_func = theano.function(inputs=[data_inputs],
                                   outputs=pk_grad,
                                   allow_input_downcast=True)

    pk_prob_func = theano.function(inputs=[data_inputs],
                                   outputs=y_hat[:, label],
                                   allow_input_downcast=True)

    return pk_prob_func, pk_grad_func


def get_logr_grad(dataset, label=2, insert_sigmoid=False):
    """
    Interface to extensions.py which asks for this function
    """
    return build_classifier_grad(dataset, label=label, insert_sigmoid=insert_sigmoid)
