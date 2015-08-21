"""
We can perturb the diffusion kernel by an arbitrary function r(x), so that we
now sample from p^tilde(x) = (1 / Z) r(x) p(x). Here, we prepare such a
function where r(x) is a classifier of the form r(x) = p(y_0|x), and we mix
in the perturbation pas r(x^t) = r(x) ** (T-t)/T """

"""
Convolutional network
Run the training of a classifier by doing:
python perturb.py --num-epochs 50 --dataset <dataset>
saves a file 'convmlp_<dataset>.zip'
that is subsequently loaded to build the gradient function.
```
Series of convolutional layers followed by a MLP
Applied to CIFAR10
Based off of Lenet code in github.com/mila-udem/blocks-examples/mnist_lenet
"""
import theano
import logging
from argparse import ArgumentParser

import numpy as np
import os

from theano import tensor
import theano.tensor as T

from blocks.algorithms import GradientDescent, AdaDelta
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Logistic)
from blocks.bricks.conv import (
    ConvolutionalLayer, ConvolutionalSequence, Flattener)
from blocks.bricks.cost import (CategoricalCrossEntropy, BinaryCrossEntropy,
                                MisclassificationRate)
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.serialization import dump, load
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.utils import named_copy
# from blocks.serialization import dump, load

from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import ScaleAndShift


class ConvMLP(FeedforwardSequence, Initializable):

    """LeNet-like convolutional network.

    The class implements LeNet, which is a convolutional sequence with
    an MLP on top (several fully-connected layers). For details see
    [LeCun95]_.

    .. [LeCun95] LeCun, Yann, et al.
       *Comparison of learning algorithms for handwritten digit
       recognition.*,
       International conference on artificial neural networks. Vol. 60.

    Parameters
    ----------
    conv_activations : list of :class:`.Brick`
        Activations for convolutional network.
    num_channels : int
        Number of channels in the input image.
    image_shape : tuple
        Input image shape.
    filter_sizes : list of tuples
        Filter sizes of :class:`.blocks.conv.ConvolutionalLayer`.
    feature_maps : list
        Number of filters for each of convolutions.
    pooling_sizes : list of tuples
        Sizes of max pooling for each convolutional layer.
    top_mlp_activations : list of :class:`.blocks.bricks.Activation`
        List of activations for the top MLP.
    top_mlp_dims : list
        Numbers of hidden units and the output dimension of the top MLP.
    conv_step : tuples
        Step of convolution (similar for all layers).
    border_mode : str
        Border mode of convolution (similar for all layers).

    """

    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims,
                 conv_step=None, border_mode='valid', **kwargs):
        if conv_step is None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        parameters = zip(conv_activations, filter_sizes, feature_maps,
                         pooling_sizes)

        # Construct convolutional layers with corresponding parameters
        self.layers = [ConvolutionalLayer(filter_size=filter_size,
                                          num_filters=num_filter,
                                          pooling_size=pooling_size,
                                          activation=activation.apply,
                                          conv_step=self.conv_step,
                                          border_mode=self.border_mode,
                                          name='conv_pool_{}'.format(i))
                       for i, (activation, filter_size, num_filter,
                               pooling_size)
                       in enumerate(parameters)]
        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()
        application_methods = [self.conv_sequence.apply, self.flattener.apply,
                               self.top_mlp.apply]
        super(ConvMLP, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [np.prod(conv_out_dim)] + self.top_mlp_dims


def train(save_to, num_epochs, feature_maps=None, mlp_hiddens=None,
          conv_sizes=None, pool_sizes=None, batch_size=500, dataset='MNIST'):

    print 'The dataset is ' + args.dataset
    # Initialize the training set
    if dataset == 'MNIST':
        from fuel.datasets import MNIST
        dataset_train = MNIST(['train'])
        dataset_test = MNIST(['test'])
        n_colors = 1
        spatial_width = 28
    elif dataset == 'CIFAR10':
        from fuel.datasets import CIFAR10
        dataset_train = CIFAR10(['train'])
        dataset_test = CIFAR10(['test'])
        n_colors = 3
        spatial_width = 32
    elif dataset == 'IMAGENET':
        from imagenet_data import IMAGENET
        spatial_width = 128
        dataset_train = IMAGENET(['train'], width=spatial_width)
        dataset_test = IMAGENET(['test'], width=spatial_width)
        n_colors = 3
    else:
        raise ValueError("Unknown dataset %s." % args.dataset)

    train = dataset_train
    test = dataset_test

    train_stream = DataStream.default_stream(
        train, iteration_scheme=ShuffledScheme(
            train.num_examples, batch_size))

    test_stream = DataStream.default_stream(
        test,
        iteration_scheme=ShuffledScheme(
            test.num_examples, batch_size))

    # make the training data 0 mean and variance 1
    # TODO compute mean and variance on full dataset, not minibatch
    Xbatch = next(train_stream.get_epoch_iterator())[0]
    scl = (
        1. / np.sqrt(np.mean((Xbatch - np.mean(Xbatch))**2))).astype('float32')
    shft = (-np.mean(Xbatch * scl)).astype('float32')
    # scale is applied before shift
    train_stream = ScaleAndShift(
        train_stream, scl, shft, which_sources=('features',))
    test_stream = ScaleAndShift(
        test_stream, scl, shft, which_sources=('features',))

    # ConvMLP Parameters
    image_size = (spatial_width, spatial_width)
    num_channels = n_colors
    num_conv = 3  # Number of Convolutional Layers
    if feature_maps is None:
        feature_maps = [20, 30, 30]
        if not len(feature_maps) == num_conv:
            raise ValueError('Must specify more feature maps')
    if conv_sizes is None:
        conv_sizes = [5] * num_conv
    if pool_sizes is None:
        pool_sizes = [2] * num_conv
    if mlp_hiddens is None:
        mlp_hiddens = [500]
    output_size = 10

    # Use ReLUs everywhere and softmax for the final prediction
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Logistic()]
    convnet = ConvMLP(conv_activations, num_channels, image_size,
                      filter_sizes=zip(conv_sizes, conv_sizes),
                      feature_maps=feature_maps,
                      pooling_sizes=zip(pool_sizes, pool_sizes),
                      top_mlp_activations=mlp_activations,
                      top_mlp_dims=mlp_hiddens + [output_size],
                      border_mode='full',
                      weights_init=Uniform(width=.2),
                      biases_init=Constant(0))

    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    convnet.push_initialization_config()
    for i in range(num_conv):
        convnet.layers[i].weights_init = Uniform(width=.2)
    convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    convnet.initialize()
    logging.info("Input dim: {} {} {}".format(
        *convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        logging.info("Layer {} dim: {} {} {}".format(
            i, *layer.get_dim('output')))

    x = tensor.tensor4('features')
    y = tensor.lmatrix('targets')

    # Normalize input and apply the convnet
    probs = convnet.apply(x)

    def binary_ce_oneshot(y, y_hat):
        """
        Returns cost for binary cross entropy

        ----------
        Parameters
        ----------

        y : theano.tensor.lmatrix (n_bat, 1)
            Target labels as an array of integers
        y_hat : theano.tensor.matrix (n_bat, n_classes)
            Logistic estimator for y (i.e. train independent classifiers)

        -------
        Returns
        -------

        cost : theano scalar
            Binary Cross entropy after converting each entry of y to a
            one hot vector
        """
        y_onehot = T.eye(y_hat.shape[1])[y.flatten()]
        cost = BinaryCrossEntropy().apply(y_onehot, y_hat)
        return cost

    cost = named_copy(binary_ce_oneshot(y, probs), 'cost')

#    cost = named_copy(CategoricalCrossEntropy().apply(y.flatten(),
#                                                      probs), 'cost')
    error_rate = named_copy(MisclassificationRate().apply(y.flatten(), probs),
                            'error_rate')

    cg = ComputationGraph([cost, error_rate])

    # Apply Dropout to outputs of rectifiers
    from blocks.roles import OUTPUT
    vs = VariableFilter(roles=[OUTPUT])(cg.variables)
    vs1 = [v for v in vs if v.name.startswith('rectifier')]
    vs1 = vs1[0: -2]  # Only first two layers
    cg = apply_dropout(cg, vs1, 0.5)

    # Train with AdaDelta
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=AdaDelta())

    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      test_stream,
                      prefix="test"),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  ProgressBar(),
                  Printing()]

    model = Model(cost)

    main_loop = MainLoop(algorithm, train_stream, model=model,
                         extensions=extensions)

    main_loop.run()
    model_dir = 'models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    classifier_fn = os.path.join(model_dir, dataset + '_classifier.zip')
    with open(classifier_fn, 'w') as f:
        dump(convnet, f)


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

    FIXME: probably the case that you need to load in the relevant bricks
            modules to open the classifier file
    """
    classifier_fn = 'models/' + dataset + '_classifier.zip'
    with open(classifier_fn, 'r') as f:
        classifier_brick = load(f)

    x = theano.tensor.tensor4('features')
    y_hat = classifier_brick.apply(x)

    pk_grad = T.sum(T.log(y_hat[:, label]))  # Trick to get dy[i]/dx[i]
    pk_grad_func = theano.function(inputs=[x],
                                   outputs=pk_grad,
                                   allow_input_downcast=True)

    # Note y_hat vectorized giving an output shaped (batches, labels),
#    pk_grad = theano.gradient.jacobian(tensor.log(y_hat[:, label]), x)
    # FIXME: does dy[i]/dx[j] instead of dy[i]/dx[i]

#    pk_grad_func1 = theano.function(inputs=[x],
#                                    outputs=pk_grad,
#                                    allow_input_downcast=True)
#
#    def pk_grad_func(x):
#        """
#        Takes diagonal of first two terms of derivative
#        """
#        res = pk_grad_func1(x)
#        n_s = res.shape[0]
#        di = np.diag_indices(n_s)
#        return res[di]

    pk_prob_func = theano.function(inputs=[x],
                                   outputs=y_hat[:, label],
                                   allow_input_downcast=True)

    return pk_prob_func, pk_grad_func


def get_logr_grad(dataset, label=2):
    """
    Interface to extensions.py which asks for this function
    """
    return build_classifier_grad(dataset, label=label)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network ")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="classifier_model.pkl", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument("--feature-maps", type=int, nargs='+',
                        default=[32, 32, 32],
                        help="List of feature maps numbers.")
    parser.add_argument("--mlp-hiddens", type=int, nargs='+', default=[64],
                        help="List of numbers of hidden units for the MLP.")
    parser.add_argument("--conv-sizes", type=int, nargs='+', default=[5, 5, 5],
                        help="Convolutional kernels sizes. The kernels are "
                        "always square.")
    parser.add_argument("--pool-sizes", type=int, nargs='+', default=[2, 1, 1],
                        help="Pooling sizes. The pooling windows are always "
                             "square. Should be the same length as "
                             "--conv-sizes.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size.")
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Name of dataset to use.')
    args = parser.parse_args()
    train(**vars(args))
