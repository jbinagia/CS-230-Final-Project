import numpy as np
import tensorflow as tf

from .system import System
from ..utils import rotation_matrix

#################################################################################

class NematicLattice(System):
    """
    A 3-dimensional nematic lattice with vector units on each lattice site.
    """

    params_default = {
        "h" : 0.0,
        "J" : 1.0
    }

    def __init__(self, params = None, **kwargs):
        # Init parent class
        super().__init__(params, **kwargs)
        self.field = np.array([0.0, 0.0, 1.0]) # Nematic field always points to z, coupling value `h` dictates contribution

    def init_coords(self, N, **kwargs):
        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        shape = (N, N, N, 3)
        x = 2.0*np.random.rand(*shape) - 1
        norms = np.linalg.norm(x, axis = 3)

        # Return normalized vectors of length 1
        return (x.reshape(np.prod(x.shape[:-1]), -1) / norms.reshape(norms.size, 1)).reshape(x.shape)

    def energy(self, x):
        N = x.shape[0]
        en = 0.0    
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    s = x[i,j,k]
                    nb = self._neighbor_sum(x, i, j, k)
                    en += -0.25*self.params["J"]*np.dot(nb, s)
                    en += -self.params["h"]*np.dot(self.field, s)

        return en

    def energy_idx(self, x, idx):
        N = x.shape[0]
        i, j, k = np.unravel_index(idx, (N, N, N))

        s = x[i,j,k]
        nb = self._neighbor_sum(x, i, j, k)
        en = -0.5*self.params["J"]*np.dot(nb, s)
        en += -self.params["h"]*np.dot(self.field, s)

        return en

    def step(self, x, **kwargs):
        theta = kwargs.get("theta", np.pi/2)

        N = x.shape[0]
        i, j, k = np.random.randint(N, size = 3)
        idx = np.ravel_multi_index((i, j, k), (N, N, N))

        M = rotation_matrix(np.random.rand(3), np.random.rand()*theta)
        new = np.copy(x)
        new[i, j, k, :] = np.dot(M, x[i, j, k])

        return idx, new

    def oprm(self, x):
        """Order parameter for a NematicLattice is the nematic parameter, S = 3*<sz>/2 - 1/2"""
        # np.abs so it measures magnitude of alignment w/ field
        sig_z = np.mean(x[:,:,:,2])
        S = 3 * sig_z / 2 - 0.5 # Nematic order parameter
        return S

    def num_sites(self, x):
        N = x.shape[0]
        return N**3

    def _neighbor_sum(self, x, i, j, k):
        N = x.shape[0]
        return x[(i+1)%N, j, k] + x[(i-1)%N, j, k] \
             + x[i, (j+1)%N, k] + x[i, (j-1)%N, k] \
             + x[i, j, (k+1)%N] + x[i, j, (k-1)%N]

    #################################################################################

    def draw_config(self, x, alpha = 0.75, figsize = (6, 6)):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D 

        fig = plt.figure(figsize = figsize)
        ax = fig.gca(projection = '3d')
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        ax.set_zlabel("x")

        N = len(x)
        gx, gy, gz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
        u, v, w = x.T

        # How to color 3D vectors (can alter cmap for variations)
        # Color by azimuthal angle
        c = np.arctan2(v, u)
        # Flatten and normalize
        c = (c.ravel() - c.min()) / c.ptp()
        # Repeat for each body line and two head lines
        c = np.concatenate((c, np.repeat(c, 2)))
        # Colormap
        c = plt.cm.hsv(c)

        ax.quiver(gz, gy, gx, u, w, v, 
            colors = c, length = 0.5, 
            alpha = alpha, pivot = 'middle'
        )