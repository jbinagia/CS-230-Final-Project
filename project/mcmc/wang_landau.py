import numpy as np

#################################################################################

class WangLandauSampler(object):

    def __init__(self, model, x0, oprm_range, oprm_bins = 100, oprm_dE = 0.5, temperature = 1.0, 
        burnin = 0, stride = 1, mapper = None, **kwargs):
        """Metropolis Monte-Carlo simulation.

        Parameters
        ----------
        model : Energy model
            Energy model object, must provide the function energy(x) and energy_idx(x)
        x0 : [array]
            Initial configuration
        temperatures : float or array
            Temperature. By default (1.0) the energy is interpreted in reduced units.
            When given an array, its length must correspond to nwalkers, then the walkers
            are simulated at different temperatures.
        burnin : int
            Number of burn-in steps that will not be saved
        stride : int
            Every so many steps will be saved
        nwalkers : int
            Number of parallel walkers
        mapper : Mapper object
            Object with function map(X), e.g. to remove permutation.
            If given will be applied to each accepted configuration.

        """
        self.model = model
        self.temperature = temperature
        self.burnin = burnin
        self.stride = stride
        self.kwargs = kwargs

        if mapper is None:
            class DummyMapper(object):
                def map(self, X):
                    return X
            mapper = DummyMapper()
        self.mapper = mapper

        self.reset(x0)


    def _proposal_step(self):
        # Proposal step

        self.idx_prop, self.x_prop = seld.model.step(self.x, **self.kwargs)
        
        self.E_idx = self.model.energy_idx(self.x, self.idx_prop)
        self.E_idx_prop = self.model.energy_idx(self.x_prop, self.idx_prop)

    def _acceptance_step(self):
        # Acceptance step
        dE = self.E_idx_prop - self.E_idx
        acc = -np.log(np.random.rand()) > dE / self.temperature

        for i in range(len(acc)):
            self.x[i] = self.x_prop[i] if acc[i] else self.x[i]
        self.E = self.E + np.where(acc, dE, 0.0)

    def reset(self, x0):
        # Counters
        self.step = 0
        self.steps_ = []
        self.traj_ = []
        self.etraj_ = []

        # Initial configuration
        self.x = x0
        self.x = self.mapper.map(self.x)
        self.E = self.model.energy(self.x)

        # Save first frame if no burnin
        if self.burnin == 0:
            self.steps_.append(0)
            self.traj_.append(self.x)
            self.etraj_.append(self.E / self.model.num_sites(self.x))

    @property
    def steps(self):
        return np.array(self.steps_)

    @property
    def traj(self):
        return np.array(self.traj_)

    @property
    def etraj(self):
        return np.array(self.etraj_)

    def run(self, nsteps = 1, verbose = 0):
        for i in range(nsteps):
            self._proposal_step()
            self._acceptance_step()
            self.step += 1
            if verbose > 0 and i % verbose == 0:
                print('Step', i, '/', nsteps)
            if self.step > 0 and self.step > self.burnin and self.step % self.stride == 0:
                self.steps_.append(self.step)
                self.traj_.append(np.copy(self.x))
                self.etraj_.append(self.E / self.model.num_sites(self.x[0])) # Per-site energy trajectory