import numpy as np

from .system import System
from .pbc import Box

#################################################################################

def _lj_pot(r, sig, eps):
    r6 = (sig / r) ** 6
    r12 = r6 * r6
    return 4.0 * eps * (r12 - r6)

class LJFluid(System):
    """
    A Lennard-Jones fluid in a periodic simulation box.
    """

    params_default = {
        "sig" : 1.0,
        "eps" : 1.0
    }

    def __init__(self, N, L, dim = 3, params = None, **kwargs):
        # Init super coordinate array
        super().__init__(N, dim = dim)

        # Add the useful params
        if params is None:
            params = self.__class__.params_default

        self.params = params
        self.sig = params["sig"]
        self.eps = params["eps"]

        # Construct a box
        self.box = Box([L] * dim)

        # Setup the coordinates
        if (self.N * (self.sig**self.dim)) > 0.8 * self.box.volume():
            raise ValueError("Particle volume exceeds simulation box size -- try increasing your box size.")
        self._init_coords(**kwargs)

        # Do initial energy methods
        self.calc_energy()
        for idx in range(N):
            self.eidx[idx] = self.calc_energy_idx(idx)

    def _init_coords(self, **kwargs):
        init_try = kwargs.get("init_try", 100)

        self.x[0, :] = self.box.random_position()
        for i in range(1, self.N):
            success = False
            for trial in range(init_try):
                new = self.box.random_position()

                dists = self.box.distance(new - self.x[:i])
                if not np.any(dists < 0.85 * self.sig):
                    self.x[i, :] = new
                    success = True
                    break

            if not success:
                raise RuntimeError("Coordinate initialization failure " \
                    + "-- likely too many particles, or too small of a box.")

    def calc_energy(self):
        en = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                rij = self.box.distance(self.x[i] - self.x[j])
                en += _lj_pot(rij, self.sig, self.eps)
        self.en = en
        return en

    def calc_energy_idx(self, idx):
        en = 0.0
        for i in range(self.N):
            if i == idx:
                continue
            else:
                rij = self.box.distance(self.x[i] - self.x[idx])
                en += _lj_pot(rij, self.sig, self.eps)
        return en

    def displace(self, idx, **kwargs):
        step = kwargs.get("step", 0.5 * self.sig)
        return self.box.wrap(self.x[idx] + np.random.rand(self.dim) * step)