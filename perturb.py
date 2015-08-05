"""
We can perturb the diffusion kernel by an arbitrary function r(x), so that
we now sample from p^tilde(x) = (1 / Z) r(x) p(x). Here, we prepare such a function
where r(x) is a classifier of the form r(x) = p(y_0|x)
"""

"""
Convolutional network
Run the training for 50 epochs with
python perturb.py --num-epochs 50
```
Series of convolutional layers followed by a MLP
Applied to CIFAR10
Based off of Lenet code in github.com/mila-udem/blocks-examples/mnist_lenet
"""
import logging
import numpy
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale, AdaDelta
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Softmax, Activation)
from blocks.bricks.conv import (
    ConvolutionalLayer, ConvolutionalSequence, Flattener)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
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
from blocks.serialization import dump, load

from fuel.datasets import CIFAR10
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream


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
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp_dims


def train(save_to, num_epochs, feature_maps=None, mlp_hiddens=None,
         conv_sizes=None, pool_sizes=None, batch_size=500):

    # Initialize the training set
    train = CIFAR10(("train",))
    train_stream = DataStream.default_stream(
        train, iteration_scheme=ShuffledScheme(
            train.num_examples, batch_size))

    test = CIFAR10(("test",))
    test_stream = DataStream.default_stream(
        test,
        iteration_scheme=ShuffledScheme(
            test.num_examples, batch_size))

    # ConvMLP Parameters
    image_size = (32, 32)
    num_channels = 3
    num_conv = 3 # Number of Convolutional Layers
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
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
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
    cost = named_copy(CategoricalCrossEntropy().apply(y.flatten(),
                      probs), 'cost')
    error_rate = named_copy(MisclassificationRate().apply(y.flatten(), probs),
                            'error_rate')

    cg = ComputationGraph([cost, error_rate])

    # Apply Dropout to outputs of rectifiers
    from blocks.roles import OUTPUT
    vs = VariableFilter(roles=[OUTPUT])(cg.variables)
    vs1 = [v for v in vs if v.name.startswith('rectifier')]
    vs1 = vs1[0: -2] # Only first two layers
    cg = apply_dropout(cg, vs1, 0.5)

    # Train with simple SGD
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

    main_loop = MainLoop(
        algorithm,
        train_stream,
        model=model,
        extensions=extensions)

    main_loop.run()
    classifier_fn = 'convmlp_cifar10.zip'
    with open(classifier_fn, 'w') as f:
        dump(convnet, f)    



def build_classifier_grad(classifier_fn='mlp.zip', label=2):
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
    with open(classifier_fn, 'r') as f:
        classifier_brick = load(f)

    x = T.tensor4('features')
    y_hat = classifier_brick.apply(x)
    
    # Note y_hat vectorized giving an output shaped (batches, labels), 
    pk_grad = theano.gradient.jacobian(tensor.log(y_hat[:, label]), x)
    # should make this more efficient using scan.. does dy[i]/dx[j]

    pk_grad_func1 = theano.function(inputs=[x],
                                   outputs=pk_grad) 

    def pk_grad_func(x):
        """
        Takes diagonal of first two terms of derivative
        """
        res = pk_grad_func1(x)
        n_s = res.shape[0]
        di = numpy.diag_indices(n_s)
        return res[di]
    
    pk_prob_func = theano.function(inputs=[x],
                                   outputs=y_hat[:, label])

    return pk_prob_func, pk_grad_func

"""
    if args.dataset == 'mnist':
        width = 28
        n_colors = 1
        n_labels = 10
    elif args.dataset == 'cifar10':
        width = 32
        n_colors = 3
        n_labels = 10
"""

    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network "
                            "on CIFAR10.")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="classifier_model.pkl", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument("--feature-maps", type=int, nargs='+',
                        default=[32, 32, 64], help="List of feature maps numbers.")
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
    args = parser.parse_args()
    train(**vars(args))
