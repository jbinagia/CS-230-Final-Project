import numpy as np
import tensorflow as tf
import keras
from deep_boltzmann.networks import IndexLayer, connect

def split_merge_indices(ndim, nchannels=2, channels=None):
    if channels is None:
        channels = np.tile(np.arange(nchannels), int(ndim/nchannels)+1)[:ndim]
    else:
        channels = np.array(channels)
        nchannels = np.max(channels) + 1
    indices_split = []
    idx = np.arange(ndim)
    for c in range(nchannels):
        isplit = np.where(channels == c)[0]
        indices_split.append(isplit)
    indices_merge = np.concatenate(indices_split).argsort()
    return channels, indices_split, indices_merge

class Permute(object):
    def __init__(self, ndim, order=None):
        """ Permutes dimensions

        Parameters:
        -----------
        order : None or array
            If None, a random permutation will be chosen.
            Otherwise, specify the new order of dimensions for x -> z.

        """
        self.ndim = ndim
        if order is None:
            order = np.random.choice(ndim, ndim, replace=False)
        self.order = order
        self.reverse = np.argsort(order)

    @classmethod
    def from_dict(cls, D):
        ndim = D['ndim']
        order = D['order']
        return cls(ndim, order=order)

    def to_dict(self):
        D = {}
        D['ndim'] = self.ndim
        D['order'] = self.order
        return D

    def connect_xz(self, x):
        self.output_z = IndexLayer(self.order)(x)
        return self.output_z

    def connect_zx(self, z):
        self.output_x = IndexLayer(self.reverse)(z)
        return self.output_x


class SplitChannels(object):
    def __init__(self, ndim, nchannels=2, channels=None):
        """ Splits channels forward and merges them backward """
        self.channels, self.indices_split, self.indices_merge = split_merge_indices(ndim, nchannels=nchannels,
                                                                                    channels=channels)

    @classmethod
    def from_dict(cls, D):
        channels = D['channels']
        dim = channels.size
        nchannels = channels.max() + 1
        return cls(dim, nchannels=nchannels, channels=channels)

    def to_dict(self):
        D = {}
        D['channels'] = self.channels
        return D

    def connect_xz(self, x):
        # split X into different coordinate channels
        self.output_z = [IndexLayer(isplit)(x) for isplit in self.indices_split]
        return self.output_z

    def connect_zx(self, z):
        # first concatenate
        x_scrambled = keras.layers.Concatenate()(z)
        # unscramble x
        self.output_x = IndexLayer(self.indices_merge)(x_scrambled) # , name='output_x'
        return self.output_x


class MergeChannels(SplitChannels):
    def connect_xz(self, x):
        # first concatenate
        z_scrambled = keras.layers.Concatenate()(x)
        # unscramble x
        self.output_z = IndexLayer(self.indices_merge)(z_scrambled) # , name='output_z'
        return self.output_z

    def connect_zx(self, z):
        # split X into different coordinate channels
        self.output_x = [IndexLayer(isplit)(z) for isplit in self.indices_split]
        return self.output_x


class Scaling(object):
    def __init__(self, ndim, scaling_factors=None, trainable=True, name_xz=None, name_zx=None):
        """ Invertible Scaling layer

        Parameters
        ----------
        ndim : int
            Number of dimensions
        scaling_factors : array
            Initial scaling factors, must be of shape (1, ndim)
        trainable : bool
            If True, scaling factors are trainable. If false, they are fixed
        name_xz : str
            Name for Sxz
        name_xz : str
            Name for Szx

        """
        # define local classes
        class ScalingLayer(keras.engine.Layer):
            def __init__(self, log_scaling_factors, **kwargs):
                """ Layer that scales dimensions with trainable factors

                Parameters
                ----------
                scaling_factors : (1xd) array
                    scaling factors applied to columns of batch matrix.

                """
                self.log_scaling_factors = log_scaling_factors
                super().__init__(**kwargs)

            def build(self, input_shape):
                # Make weight trainable
                if self.trainable:
                    self._trainable_weights.append(self.log_scaling_factors)
                super().build(input_shape)  # Be sure to call this at the end

            def compute_output_shape(self, input_shape):
                return (input_shape[0], self.log_scaling_factors.shape[1])

        class ScalingXZ(ScalingLayer):
            def __init__(self, log_scaling_factors, **kwargs):
                """ Layer that scales the batch X in (B,d) by X * S where S=diag(s1,...,sd)
                """
                super().__init__(log_scaling_factors, **kwargs)

            def call(self, x):
                return x * tf.exp(self.log_scaling_factors)

        class ScalingZX(ScalingLayer):
            def __init__(self, log_scaling_factors, **kwargs):
                """ Layer that scales the batch X in (B,d) by X * S^(-1) where S=diag(s1,...,sd)
                """
                super().__init__(log_scaling_factors, **kwargs)

            def call(self, x):
                return x * tf.exp(-self.log_scaling_factors)

        # initialize scaling factors
        if scaling_factors is None:
            self.log_scaling_factors = keras.backend.variable(np.zeros((1, ndim)),
                                                              dtype=keras.backend.floatx(),
                                                              name='log_scale')
        else:
            self.log_scaling_factors = keras.backend.variable(np.log(scaling_factors),
                                                              dtype=keras.backend.floatx(),
                                                              name='log_scale')

        self.trainable = trainable
        self.Sxz = ScalingXZ(self.log_scaling_factors, trainable=trainable, name=name_xz)
        self.Szx = ScalingZX(self.log_scaling_factors, trainable=trainable, name=name_zx)

    @property
    def scaling_factors(self):
        return tf.exp(self.log_scaling_factors)

    @classmethod
    def from_dict(cls, D):
        scaling_factors = D['scaling_factors']
        dim = scaling_factors.shape[1]
        trainable = D['trainable']
        name_xz = D['name_xz']
        name_zx = D['name_zx']
        return Scaling(dim, scaling_factors=scaling_factors, trainable=trainable, name_xz=name_xz, name_zx=name_zx)

    def to_dict(self):
        D = {}
        D['scaling_factors'] = keras.backend.eval(self.scaling_factors)
        D['trainable'] = self.trainable
        D['name_xz'] = self.Sxz.name
        D['name_zx'] = self.Szx.name
        return D

    def connect_xz(self, x):
        def lambda_Jxz(x):
            J = tf.reduce_sum(self.log_scaling_factors, axis=1)[0]
            return J * keras.backend.ones((tf.shape(x)[0], 1))
        self.log_det_xz = keras.layers.Lambda(lambda_Jxz)(x)
        z = self.Sxz(x)
        return z

    def connect_zx(self, z):
        def lambda_Jzx(x):
            J = tf.reduce_sum(-self.log_scaling_factors, axis=1)[0]
            return J * keras.backend.ones((tf.shape(x)[0], 1))
        self.log_det_zx = keras.layers.Lambda(lambda_Jzx)(z)
        x = self.Szx(z)
        return x

    @property
    def log_det_Jxz(self):
        """ Log of |det(dz/dx)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_xz

    @property
    def log_det_Jzx(self):
        """ Log of |det(dx/dz)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_zx


class CompositeLayer(object):
    def __init__(self, transforms):
        """ Composite layer consisting of multiple keras layers with shared parameters  """
        self.transforms = transforms

    @classmethod
    def from_dict(cls, d):
        from deep_boltzmann.networks.util import deserialize_layers
        transforms = deserialize_layers(d['transforms'])
        return cls(transforms)

    def to_dict(self):
        from deep_boltzmann.networks.util import serialize_layers
        D = {}
        D['transforms'] = serialize_layers(self.transforms)
        return D


class NICER(CompositeLayer):
    def __init__(self, transforms):
        """ Two sequential NICE transformations and their inverse transformations.

        Parameters
        ----------
        transforms : list
            List with [M1, M2] containing the keras layers for nonlinear transformation 1 and 2.

        """
        super().__init__(transforms)
        self.M1 = transforms[0]
        self.M2 = transforms[1]

    def connect_xz(self, x):
        x1 = x[0]
        x2 = x[1]
        self.input_x1 = x1
        self.input_x2 = x2

        # first stage backward
        y2 = x2
        y1 = keras.layers.Subtract()([x1, connect(x2, self.M2)])
        # second stage backward
        z1 = y1
        z2 = keras.layers.Subtract()([y2, connect(y1, self.M1)])

        return [z1, z2] + x[2:]  # append other layers if there are any

    def connect_zx(self, z):
        z1 = z[0]
        z2 = z[1]
        self.input_z1 = z1
        self.input_z2 = z2

        # first stage forward
        y1 = z1
        y2 = keras.layers.Add()([z2, connect(z1, self.M1)])
        # second stage forward
        x2 = y2
        x1 = keras.layers.Add()([y1, connect(y2, self.M2)])

        return [x1, x2] + z[2:]  # append other layers if there are any


class RealNVP(CompositeLayer):
    def __init__(self, transforms):
        """ Two sequential NVP transformations and their inverse transformatinos.

        Parameters
        ----------
        transforms : list
            List [S1, T1, S2, T2] with keras layers for scaling and translation transforms

        """
        super().__init__(transforms)
        self.S1 = transforms[0]
        self.T1 = transforms[1]
        self.S2 = transforms[2]
        self.T2 = transforms[3]

    def connect_xz(self, x):
        def lambda_exp(x):
            return keras.backend.exp(x)
        def lambda_sum(x):
            return keras.backend.sum(x[0], axis=1, keepdims=True) + keras.backend.sum(x[1], axis=1, keepdims=True)

        x1 = x[0]
        x2 = x[1]
        self.input_x1 = x1
        self.input_x2 = x2

        y1 = x1
        self.Sxy_layer = connect(x1, self.S1)
        self.Txy_layer = connect(x1, self.T1)
        prodx = keras.layers.Multiply()([x2, keras.layers.Lambda(lambda_exp)(self.Sxy_layer)])
        y2 = keras.layers.Add()([prodx, self.Txy_layer])

        self.output_z2 = y2
        self.Syz_layer = connect(y2, self.S2)
        self.Tyz_layer = connect(y2, self.T2)
        prody = keras.layers.Multiply()([y1, keras.layers.Lambda(lambda_exp)(self.Syz_layer)])
        self.output_z1 = keras.layers.Add()([prody, self.Tyz_layer])

        # log det(dz/dx)
        self.log_det_xz = keras.layers.Lambda(lambda_sum)([self.Sxy_layer, self.Syz_layer])

        return [self.output_z1, self.output_z2] + x[2:]  # append other layers if there are any

    def connect_zx(self, z):
        def lambda_negexp(x):
            return keras.backend.exp(-x)
        def lambda_negsum(x):
            return keras.backend.sum(-x[0], axis=1, keepdims=True) + keras.backend.sum(-x[1], axis=1, keepdims=True)

        z1 = z[0]
        z2 = z[1]
        self.input_z1 = z1
        self.input_z2 = z2

        y2 = z2
        self.Szy_layer = connect(z2, self.S2)
        self.Tzy_layer = connect(z2, self.T2)
        z1_m_Tz2 = keras.layers.Subtract()([z1, self.Tzy_layer])
        y1 = keras.layers.Multiply()([z1_m_Tz2, keras.layers.Lambda(lambda_negexp)(self.Szy_layer)])

        self.output_x1 = y1
        self.Syx_layer = connect(y1, self.S1)
        self.Tyx_layer = connect(y1, self.T1)
        y2_m_Ty1 = keras.layers.Subtract()([y2, self.Tyx_layer])
        self.output_x2 = keras.layers.Multiply()([y2_m_Ty1, keras.layers.Lambda(lambda_negexp)(self.Syx_layer)])

        # log det(dx/dz)
        # TODO: check Jacobian
        self.log_det_zx = keras.layers.Lambda(lambda_negsum)([self.Szy_layer, self.Syx_layer])

        return [self.output_x1, self.output_x2] + z[2:]  # append other layers if there are any

    @property
    def log_det_Jxz(self):
        """ Log of |det(dz/dx)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_xz

    @property
    def log_det_Jzx(self):
        """ Log of |det(dx/dz)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_zx
