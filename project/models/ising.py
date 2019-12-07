import numpy as np
import torch

from .system import System

#################################################################################

class IsingModel(System):
    """
    A 2-dimensional Ising model grid.
    """

    params_default = {
        "h" : 0.0,
        "J" : 1.0,
        "E_high": 1000000,
        "E_max": 10**12 # cutoff to avoid overflow
    }

    def __init__(self, params = None, **kwargs):
        # Init parent class
        super().__init__(params, **kwargs)

    def init_coords(self, N, **kwargs):
        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        shape = [N] * 2
        return 2 * np.random.randint(0, 2, size = shape) - 1

    def energy(self, x):
        assert x.shape[0] == x.shape[1] # x must be square and not flattened

        # Non-PBCs
        neigh = torch.zeros_like(x)
        neigh[:, :-1] += x[:, 1:]
        neigh[:, 1:] += x[:, :-1]
        neigh[:-1] += x[1:]
        neigh[1:] += x[:-1]

        # Handle PBCs
        neigh[0, :] += x[-1, :]
        neigh[-1, :] += x[0, :]
        neigh[:, 0] += x[:, -1]
        neigh[:, -1] += x[:, 0]

        en_field = -self.params["h"] * torch.sum(x)
        en_pair = -0.5 * self.params["J"] * torch.sum(neigh * x)

        # regularize energy
        tmp_energy = en_field + en_pair
        # if  tmp_energy < self.params["E_high"]:
        return tmp_energy
        # elif tmp_energy  >= self.params["E_high"] and tmp_energy < self.params["E_max"]:
        #     return self.params["E_high"] + np.log(tmp_energy - self.params["E_high"] + 1)
        # elif tmp_energy > self.params["E_max"]:
        #     return self.params["E_high"] + np.log(self.params["E_max"] - self.params["E_high"] + 1)

    def energy_vec(self, x):
        # Non-PBCs
        neigh = torch.zeros_like(x)
        neigh[:, :-1] += x[:, 1:]
        neigh[:, 1:] += x[:, :-1]
        neigh[:-1] += x[1:]
        neigh[1:] += x[:-1]

        # Handle PBCs
        neigh[0, :] += x[-1, :]
        neigh[-1, :] += x[0, :]
        neigh[:, 0] += x[:, -1]
        neigh[:, -1] += x[:, 0]

        en_field = -self.params["h"] * torch.sum(x,1)
        en_pair = -0.5 * self.params["J"] * torch.sum(neigh * x,1)
        return en_field + en_pair


    def energy_idx(self, x, idx):
        N = x.shape[0]
        i, j = np.unravel_index(idx, (N, N))

        s = x[i,j]
        en_field = -self.params["h"]*s

        nb = self._neighbor_sum(x, i, j)
        en_pair = -0.5*self.params["J"]*nb*s

        return en_field + en_pair

    def step(self, x, **kwargs):
        N = x.shape[0]
        i, j = np.random.randint(N, size = 2)
        idx = np.ravel_multi_index((i, j), (N, N))

        new = np.copy(x)
        new[i,j] = -1 * x[i,j]

        return idx, new

    def oprm(self, x):
        """Order parameter for IsingModel is the average magnetization."""
        return np.mean(x)

    def num_sites(self, x):
        N = x.shape[0]
        return N**2

    def _neighbor_sum(self, x, i, j):
        N = x.shape[0]
        return x[(i+1)%N, j] + x[(i-1)%N, j] \
             + x[i, (j+1)%N] + x[i, (j-1)%N]

    #################################################################################

    def draw_config(self, x, figsize = (6, 6)):
        import matplotlib.pyplot as plt

        (fig, ax) = plt.subplots(1, figsize = figsize)

        N = x.shape[0]
        X, Y = np.meshgrid(range(N+1), range(N+1))

        plt.setp(ax.get_yticklabels(), visible = False)
        plt.setp(ax.get_xticklabels(), visible = False)
        plt.pcolormesh(X, Y, -1*x, cmap = plt.cm.RdBu)
