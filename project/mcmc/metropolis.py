import numpy as np

#################################################################################

class MetropolisSampler(object):

    def __init__(self, model, x0, temperature = 1.0, burnin = 0, stride = 1, nwalkers = 1, 
        mapper = None, **kwargs):
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
        self.nwalkers = nwalkers
        self.kwargs = kwargs

        if mapper is None:
            class DummyMapper(object):
                def map(self, X):
                    return X
            mapper = DummyMapper()
        self.mapper = mapper

        self.reset(x0)

        self.dog = []

    def _proposal_step(self):
        # Proposal step

        props = [self.model.step(xi, **self.kwargs) for xi in self.x]
        self.idx_prop = np.array([p[0] for p in props])
        self.x_prop = np.array([p[1] for p in props])
        
        self.E_idx = np.array([self.model.energy_idx(xi, i) for (xi, i) in zip(self.x, self.idx_prop)])
        self.E_idx_prop = np.array([self.model.energy_idx(xi, i) for (xi, i) in zip(self.x_prop, self.idx_prop)])

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
        self.x = np.tile(x0, tuple([self.nwalkers]) + tuple([1]) * len(x0.shape) )
        self.x = self.mapper.map(self.x)
        self.E = np.array([self.model.energy(xi) for xi in self.x])

        # Save first frame if no burnin
        if self.burnin == 0:
            self.steps_.append(0)
            self.traj_.append(self.x)
            self.etraj_.append(self.E / self.model.num_sites(self.x[0]))

    @property
    def steps(self):
        return np.array(self.steps_)

    @property
    def trajs(self):
        """Returns a list of trajectories, one trajectory for each walker."""
        T = np.array(self.traj_).astype(np.float32)
        return [T[:, i, :] for i in range(T.shape[1])]

    @property
    def traj(self):
        return self.trajs[0]

    @property
    def etrajs(self):
        """Returns a list of energy trajectories, one trajectory for each walker."""
        E = np.array(self.etraj_)
        return [E[:, i] for i in range(E.shape[1])]

    @property
    def etraj(self):
        return self.etrajs[0]

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


class ReplicaMetropolisSampler(object):

    def __init__(self, model, x0, temperatures, burnin = 0, stride = 1, mapper = None, **kwargs):
        if temperatures.size % 2 == 0:
            raise ValueError('Please use an odd number of temperatures.')

        self.toggle = 0
        self.temperatures = temperatures
        self.sampler = MetropolisSampler(model, x0, temperature = temperatures,
            burnin = burnin, stride = stride, nwalkers = temperatures.size, mapper = mapper, **kwargs)

    @property
    def steps(self):
        splits = np.split(self.sampler.steps, 
            1 + np.argwhere(self.sampler.steps[1:] < self.sampler.steps[:-1]).flatten())

        steps = splits[0]
        for i in range(1, len(splits)): 
            steps = np.append(steps, splits[i-1][-1] + splits[i])
        return steps

    @property
    def trajs(self):
        return self.sampler.trajs

    @property
    def etrajs(self):
        return self.sampler.etrajs

    def run(self, nepochs = 1, nsteps_per_epoch = 1, verbose = 0):
        for i in range(nepochs):
            self.sampler.run(nsteps = nsteps_per_epoch)
            # Exchange
            for k in range(self.toggle, self.temperatures.size-1, 2):
                c = -(self.sampler.E[k+1] - self.sampler.E[k]) * (1.0/self.temperatures[k+1] - 1.0/self.temperatures[k])
                acc = -np.log(np.random.rand()) > c
                if acc:
                    h = self.sampler.x[k].copy()
                    self.sampler.x[k] = self.sampler.x[k+1].copy()
                    self.sampler.x[k+1] = h
                    h = self.sampler.E[k]
                    self.sampler.E[k] = self.sampler.E[k+1]
                    self.sampler.E[k+1] = h
            self.toggle = 1 - self.toggle