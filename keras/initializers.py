from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import six
from . import backend as K
from .utils.generic_utils import serialize_keras_object
from .utils.generic_utils import deserialize_keras_object


class Initializer(object):
    """Initializer base class: all initializers inherit from this class.
    """

    def __call__(self, shape, dtype=None):
        raise NotImplementedError

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Constant(Initializer):
    def __init__(self, value):
        self.value = value

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.constant(self.value, shape=shape, dtype=dtype)
    
    def get_config(self):
        return {'value': self.value}

class RandomNormal(Initializer):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.random_normal(shape, self.mean, self.std, dtype=dtype)

    def get_config(self):
        return {'mean': self.mean, 'std': self.std}

class RandomUniform(Initializer):
    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.random_uniform(shape, self.minval, self.maxval, dtype=dtype)

    def get_config(self):
       return {'minval': self.minval, 'maxval': self.maxval}

class TruncatedNormal(Initializer):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.truncated_normal(shape, self.mean, self.std, dtype=dtype)

    def get_config(self):
        return {'mean': self.mean, 'std': self.std}

class Orthogonal(Initializer):
    def __init__(self, gain):
        self.gain = gain

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.orthogonal_initializer(gain=self.gain)(shape, dtype=dtype)

    def get_config(self):
        return {'gain': self.gain}

class VarianceScaling(Initializer):
    """Initializer capable of adapting its scale to the shape of weights.

    With `distribution="normal"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"

    With `distribution="uniform"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

    # Arguments
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "normal", "uniform".
        seed: A Python integer. Used to seed the random generator.

    # Raises
        ValueError: In case of an invalid value for the "scale", mode" or
          "distribution" arguments.
    """

    def __init__(self, scale=1.0,
                 mode='fan_in',
                 distribution='normal',
                 seed=None):
        if scale <= 0.:
            raise ValueError('`scale` must be a positive float. Got:', scale)
        mode = mode.lower()
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument: '
                             'expected on of {"fan_in", "fan_out", "fan_avg"} '
                             'but got', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype=None, **kwargs):
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self.distribution == 'normal':
            stddev = np.sqrt(scale)
            return tf.truncated_normal(shape, 0.0, stddev,
                                      dtype=dtype, seed=self.seed)
        else:
            limit = np.sqrt(3. * scale)
            return tf.random_uniform(shape, -limit, limit,
                                    dtype=dtype, seed=self.seed)

    def get_config(self):
        return {
            'scale': self.scale,
            'mode': self.mode,
            'distribution': self.distribution,
            'seed': self.seed
        }

zero = zeros = Constant(0.0)
one = ones = Constant(1.0)
normal = RandomNormal(0, 0.05)
uniform = RandomUniform(-0.05, 0.05)
truncated_normal = TruncatedNormal(0, 0.05)
orthogonal = Orthogonal(gain=1.0)

lecun_uniform = VarianceScaling(scale=1.,
                           mode='fan_in',
                           distribution='uniform')


glorot_normal = VarianceScaling(scale=1.,
                           mode='fan_avg',
                           distribution='normal')


glorot_uniform = VarianceScaling(scale=1.,
                           mode='fan_avg',
                           distribution='uniform')


he_normal = VarianceScaling(scale=2.,
                           mode='fan_in',
                           distribution='normal')


he_uniform = VarianceScaling(scale=2.,
                           mode='fan_in',
                           distribution='uniform')


def _compute_fans(shape, data_format='channels_last'):
    """Computes the number of input and output units for a weight shape.

    # Arguments
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).

    # Returns
        A tuple of scalars, `(fan_in, fan_out)`.

    # Raises
        ValueError: in case of invalid `data_format` argument.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def get(identifier):
    if isinstance(identifier, six.string_types):
        # get it from the module
        return globals()[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret initializer identifier:',
                         identifier)

def serialize(initializer):
    return serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='initializer')

