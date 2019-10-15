import numpy as np

class System(object):
    """
    Parent class for a model system.
    Each subclass defines methods to evalute the energy
    as a function of internal coordinaes.
    """

    params_default = {}

    def __init__(self, params = None, **kwargs):
        # Set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params.copy()

        # Force update from kwargs
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    def init_coords(self):
        pass

    def calc_energy(self, x):
        pass

    def calc_energy_idx(self, x, idx):
        pass

    def displace(self, x, idx):
        pass

    def draw_config(self, x):
        pass