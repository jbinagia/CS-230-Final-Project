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

    # Each mode will implement, at least, the below
    
    def init_coords(self, *args, **kwargs):
        pass

    def energy(self, *args, **kwargs):
        pass

    def energy_idx(self, *args, **kwargs):
        pass

    def random_idx(self, *args, **kwargs):
        pass

    def displace(self, *args, **kwargs):
        pass

    def oprm(self, *args, **kwargs):
        pass

    def draw_config(self, *args, **kwargs):
        pass