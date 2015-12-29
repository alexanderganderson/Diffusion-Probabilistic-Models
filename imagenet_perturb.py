"""
We can perturb the diffusion kernel by an arbitrary function r(x), so that we
now sample from p^tilde(x) = (1 / Z) r(x) p(x). Here, we prepare such a
function where r(x) is a classifier of the form r(x) = p(y_0|x), and we mix
in the perturbation pas r(x^t) = r(x) ** (T-t)/T

Convolutional network from sklearn-theano
"""
import theano

import numpy as np
# import os

# from theano import tensor
import theano.tensor as T

from sklearn_theano.feature_extraction.caffe.googlenet import create_theano_expressions

import PIL.Image


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
    pk_grad = T.grad(T.sum(T.log(y_hat[:, label])), data_inputs)
    # Trick for dy[i]/dx[i]

    pk_grad_func = theano.function(inputs=[data_inputs],
                                   outputs=pk_grad,
                                   allow_input_downcast=True)

    pk_prob_func = theano.function(inputs=[data_inputs],
                                   outputs=y_hat[:, label],
                                   allow_input_downcast=True)

    return pk_prob_func, pk_grad_func


def resize(arr, d=224):
    """
    Takes in an array and resizes it as necessary

    Parameters
    ----------
    arr - array, shape (N, K, H, W)
        Input image of dimensions H, W

    Returns
    -------
    new_arr, array, shape (N, K, H, W)
        Resized image where images are now shape (d, d)
    """
    N, K, _, _ = arr.shape
    new_arr = np.zeros((N, K, d, d)).astype('float32')
    for i in range(N):
        img = arr[i].astype('float32')
        img = img.transpose(1, 2, 0)
        m1 = img.min()
        m2 = img.max()
        img = (img - m1)/(m2-m1) * 255.
        pimg = PIL.Image.fromarray(img.astype('uint8'))
        pimg = pimg.resize((d, d), PIL.Image.ANTIALIAS)
        new_arr[i] = np.array(pimg).transpose(2, 0, 1).astype('float32')
        new_arr[i] = m1 + (m2-m1) / 255. * new_arr[i]
    return new_arr


def get_logr_grad(dataset, shft, scl, spatial_width,
                  label=2, insert_sigmoid=False):
    """
    Interface to extensions.py which asks for this function
    """
    r1, logr_grad1 = build_classifier_grad(
        dataset, label=label, insert_sigmoid=insert_sigmoid)

    def r(X):
        X0 = resize((X - shft) / scl, d=224)
        return r1(X0)

    def logr_grad(X):
        X0 = resize((X - shft) / scl, d=224)
        g0 = logr_grad1(X0)

        return (1. / scl) * resize(g0, d=spatial_width)

    return r, logr_grad
