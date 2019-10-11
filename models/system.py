import numpy as np

class System(object):
    """
    Parent class for a model system.
    Each subclass defines methods to evalute the energy
    as a function of internal coordinaes.
    """

    def __init__(self, shape):
        self.x = np.zeros(shape)
        self.energy = 0.0

    def _init_coords(self):
        pass

    def calc_energy(self):
        pass

    def calc_energy_idx(self, idx):
        pass

    def displace(self, idx):
        pass