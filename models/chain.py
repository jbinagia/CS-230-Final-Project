import numpy as np
import tensorflow as tf

from .system import System
from .pbc import Box
from .potentials import lj_potential, harmonic_potential

#################################################################################

class GaussianChain(System):
    """
    A Gaussian polymer chain in a 3D simulation box.
    """

    params_default = {
        "sig" : 1.0,
        "eps" : 1.0,
        "r0"  : 1.5,
        "k"   : 5.0
    }

    def __init__(self, nmon, L, params = None, **kwargs):
        # Init parent class
        shape = (nmon, 3)
        super().__init__(shape, params)

        self.nmon = nmon
        self.box = Box([L, L, L])

        # Setup the coordinates
        if (self.nmon * (self.params["sig"]**3)) > 0.9 * self.box.volume():
            raise ValueError("Particle volume exceeds simulation box size -- try increasing your box size.")

        # Initial coordinates and energy
        self._init_coords(kwargs)
        self.calc_energy()

    def _init_coords(self, **kwargs):
        init_try = kwargs.get("init_try", 500)
        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        self.x[0, :] = self.box.random_position()
        for i in range(1, self.nmon):
            success = False
            for trial in range(init_try):
                z = np.random.randn(3)
                z *= self.params["r0"] / np.linalg.norm(z)
                new = self.box.wrap(self.x[i-1] + z)

                dists = self.box.distance(new, self.x[:i])
                if not np.any(dists < 0.85 * self.params["sig"]):
                    self.x[i, :] = new
                    success = True
                    break

            if not success:
                raise RuntimeError("Coordinate initialization failure " \
                    + "-- likely too many particles, or too small of a box.")

    def calc_energy(self):
        # Raw calculation with O(N^2) scaling, slow
        en_nb = 0.0
        for i in range(self.nmon):
            for j in range(i+1, self.nmon):
                rij = self.box.distance(self.x[i], self.x[j])
                en_nb += lj_potential(rij, self.params["sig"], self.params["eps"])

        en_b = 0.0
        for i in range(1, self.nmon):
            rij = self.box.distance(self.x[i], self.x[i-1])
            en_b += harmonic_potential(rij, self.params["r0"], self.params["k"])

        self.energy_nonbonded = en_nb
        self.energy_bonded = en_b
        self.energy = en_nb + en_b
        return self.energy

    def calc_energy_idx(self, idx):
        # For incremental MCMC updates, single particle, scales O(N)
        eidx = 0.0
        for i in range(self.nmon):
            if i == idx:
                continue
            else:
                rij = self.box.distance(self.x[idx], self.x[i])
                eidx += lj_potential(rij, self.params["sig"], self.params["eps"])
                if (i == idx-1) or (i == idx+1):
                    eidx += harmonic_potential(rij, self.params["r0"], self.params["k"])
        return eidx

    def displace(self, idx = None, **kwargs):
        step = kwargs.get("step", 0.5 * self.params["sig"])
        if idx:
            return self.box.wrap(self.x[idx] + np.random.randn(3) * step)
        else:
            return self.box.wrap(self.x + np.random.randn(*self.shape) * step)

    def draw_config(self, s = 100, c = "b"):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = Axes3D(fig)
        x, y, z = self.x[:,0], self.x[:,1], self.x[:,2]
        ax.scatter(x, y, z, c = c, s = s)
        ax.plot(x, y, z, c = c)