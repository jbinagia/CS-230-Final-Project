import numpy as np

class System(object):
    """
    Parent class for a model system.
    Each subclass defines methods to evalute the energy
    as a function of internal coordinaes.
    """

    params_default = {}

    def __init__(self, shape, params = None):
        self.shape = shape
        self.n = np.prod(shape) # Vector dimension of flattened coordinates

        # Set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params

    def x_to_vec(self):
        return self.x.flatten()

    def vec_to_x(self, vec):
        self.x = vec.reshape(self.shape)
        return self.x

    def init_coords(self):
        pass

    def calc_energy(self, x):
        pass

    def calc_energy_idx(self, x, idx):
        pass

    def displace(self, x, idx = None):
        pass