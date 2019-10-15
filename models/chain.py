import numpy as np
import tensorflow as tf

from utils import lj_potential, harmonic_potential
from .system import System
from .pbc import Box

#################################################################################

class GaussianChain(System):
    """
    A Gaussian polymer chain in a 3D simulation box.
    """

    params_default = {
        "sig" : 1.0,
        "eps" : 1.0,
        "r0"  : 1.5,
        "k"   : 5.0,
        "L"   : 5.0
    }

    def __init__(self, params = None, **kwargs):
        # Init parent class
        super().__init__(params, **kwargs)

        # Create an appropriate simulation box
        self.box = Box([self.params["L"]] * 3) # Fixed at 3D

    def init_coords(self, N, init_try = 500, **kwargs):
        # Check coordinate size
        if (N * (self.params["sig"]**3)) > 0.9 * self.box.volume():
            raise ValueError("Particle volume exceeds simulation box size -- try increasing your box size.")

        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        # Place while checking for overlap
        x = self.box.center()
        for i in range(1, N):
            success = False
            for trial in range(init_try):
                z = np.random.randn(3)
                z *= self.params["r0"] / np.linalg.norm(z)
                new = self.box.wrap(x[i-1] + z)

                dists = self.box.distance(new, x)
                if not np.any(dists < 0.85 * self.params["sig"]):
                    x = np.vstack((x, new))
                    success = True
                    break

            if not success:
                raise RuntimeError("Coordinate initialization failure " \
                    + "-- likely too many particles, or too small of a box.")

        return x

    def energy(self, x):
        # Raw calculation with O(N^2) scaling, slow
        en_nb = 0.0
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                rij = self.box.distance(x[i], x[j])
                en_nb += lj_potential(rij, self.params["sig"], self.params["eps"])

        en_b = 0.0
        for i in range(1, len(x)):
            rij = self.box.distance(x[i], x[i-1])
            en_b += harmonic_potential(rij, self.params["r0"], self.params["k"])

        return en_nb + en_b

    def energy_idx(self, x, idx):
        # For incremental MCMC updates, single particle, scales O(N)
        en = 0.0
        for i in range(len(x)):
            if i == idx:
                continue
            else:
                rij = self.box.distance(x[idx], x[i])
                en += lj_potential(rij, self.params["sig"], self.params["eps"])
                if (i == idx-1) or (i == idx+1):
                    en += harmonic_potential(rij, self.params["r0"], self.params["k"])
        return en

    def random_idx(self, x):
        return np.random.randint(x.shape[0])

    def displace(self, x, **kwargs):
        step = kwargs.get("step", 0.5 * self.params["sig"])
        return self.box.wrap(x + np.random.randn(*x.shape) * step)

    def _unwrap(self, x):
        """Unwraps a chain across periodic boundaries."""
        delta = self.box.min_image(x[1:] - x[:-1])
        return np.cumsum(np.vstack((x[0], delta)), axis = 0)

    def oprm(self, x):
        """Order parameter for a GaussianChain is radius of gyration."""
        xu = self._unwrap(x)
        com = np.mean(xu, axis = 0)
        xc = xu - com

        rg2 = np.sum(np.linalg.eigvals(
                np.mean(np.apply_along_axis(lambda x: np.outer(x, x), 1, xc), axis = 0)
            ))
        return rg2

    #################################################################################

    def draw_config(self, x, alpha = 0.75, figsize = (8, 8)):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize = figsize)
        ax = Axes3D(fig)

        sig = self.params["sig"]
        ix, iy, iz = x[:,0], x[:,1], x[:,2]
        ax.scatter(ix, iy, iz, c = 'gray', edgecolors = 'black', s = 200 * sig)
        ax.plot(ix, iy, iz, c = 'gray')