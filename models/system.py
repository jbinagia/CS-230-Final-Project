import numpy as np

class System(object):
    """
    Parent class for a model system.
    Each subclass defines methods to evalute the energy
    as a function of internal coordinaes.
    """

    def __init__(self, N, dim):
        self.N = N
        self.dim = dim
        self.x = np.zeros((N, dim))

        # Holders for the energy of the system
        # Any method that updates the system position should update these values
        self.en = 0.0
        self.eidx = np.zeros(N)

    def _init_coords(self):
        pass

    def calc_energy(self):
        pass

    def calc_energy_idx(self, idx):
        pass

    def displace(self, idx):
        pass
