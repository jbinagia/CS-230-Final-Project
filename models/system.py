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
        self.x = np.zeros(shape)
        self.energy = 0.0

        # Set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params

    def x_to_vec(self):
        return self.x.flatten()

    def vec_to_x(self, vec):
        self.x = vec.reshape(self.shape)
        return self.x

    def _init_coords(self):
        pass

    def calc_energy(self):
        pass

    def calc_energy_idx(self, idx):
        """
        Calculates the energy at coordinate `idx` in the system,
        which may correspond to a single particle or lattice site.
        """
        pass

    def displace(self, idx = None):
        pass