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


def build_classifier_grad(dataset, label=2):
    """
    Loads a classifier, and builds functions p(y_label|x) and
        d p(y_label|x)/dx where x is the image

    ----------
    Parameters
    ----------
    classifier_fn : string
         Filename to load the brick containing the classifier
    label : int
         Integer determining which class
    """
    b, x = create_theano_expressions()
    x.name = 'features'
    # b,c,0,1
    logistic_input = b['pool5/7x7_s1']
    # b, c
    # TODO: not sure if best to mean before or after logistic
    # doesn't matter is images are correct size
    li = logistic_input.mean(axis=(2, 3))

    y_hat = T.nnet.sigmoid(li)

    pk_grad = T.grad(T.sum(T.log(y_hat[:, label])), x)  # Trick for dy[i]/dx[i]
    pk_grad_func = theano.function(inputs=[x],
                                   outputs=pk_grad,
                                   allow_input_downcast=True)

    pk_prob_func = theano.function(inputs=[x],
                                   outputs=y_hat[:, label],
                                   allow_input_downcast=True)

    return pk_prob_func, pk_grad_func


def get_logr_grad(dataset, label=2):
    """
    Interface to extensions.py which asks for this function
    """
    return build_classifier_grad(dataset, label=label)
