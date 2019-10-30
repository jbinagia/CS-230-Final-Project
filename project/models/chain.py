import numpy as np

from .system import System
from .box import Box

from ..utils import lj_potential, harmonic_potential, distance_array

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
        # Faster distance array computation, self-written (can find faster versions...)
        dists = distance_array(x, x, box = self.box)
        mu, nu = np.triu_indices_from(dists, k=1)
        en_nb = np.sum(lj_potential(dists[mu, nu], self.params["sig"], self.params["eps"]))

        en_b = np.sum(
            harmonic_potential(self.box.distance(x[1:], x[:-1]), 
                self.params["r0"], self.params["k"])
        )

        return en_nb + en_b

    def energy_idx(self, x, idx):
        dists = self.box.distance(x[idx], x)
        en_nb = np.sum(lj_potential(np.delete(dists, idx), self.params["eps"], self.params["sig"]))

        if idx == 0:
            bonds = dists[1]
        elif idx == len(x) - 1:
            bonds = dists[-2]
        else:
            bonds = np.asarray([dists[idx-1], dists[idx+1]])
        en_b = np.sum(harmonic_potential(bonds, self.params["r0"], self.params["k"]))

        return en_nb + en_b

    def step(self, x, **kwargs):
        delta = kwargs.get("delta", 0.5 * self.params["sig"])

        new = np.copy(x)
        idx = np.random.randint(x.shape[0])
        new[idx, :] = self.box.wrap(x[idx] + np.random.randn(x.shape[-1]) * delta)

        return idx, new

    def oprm(self, x):
        """Order parameter for a GaussianChain is radius of gyration."""
        xu = chain._unwrap(x)
        com = np.mean(xu, axis = 0)
        xc = xu - com

        # Element-wise outer product for each position vector
        #   Performed by expaning array dimensions and element-wise mult
        outers = xc[:,:,None]*xc[:,None,:]
        return np.sum(np.linalg.eigvals(np.mean(outers, axis = 0)))

    def num_sites(self, x):
        return x.shape[0]

    def _unwrap(self, x):
        """Unwraps a chain across periodic boundaries."""
        delta = self.box.min_image(x[1:] - x[:-1])
        return np.cumsum(np.vstack((x[0], delta)), axis = 0)

    #################################################################################

    def draw_config(self, x, alpha = 0.75, figsize = (6, 6)):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize = figsize)
        ax = Axes3D(fig)

        sig = self.params["sig"]
        ix, iy, iz = x[:,0], x[:,1], x[:,2]
        ax.scatter(ix, iy, iz, c = 'gray', edgecolors = 'black', s = 200 * sig)
        ax.plot(ix, iy, iz, c = 'gray')