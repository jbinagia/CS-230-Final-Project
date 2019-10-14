import numpy as np
import tensorflow as tf
from simtk import unit
from simtk import openmm


class DoubleWellPotential(object):
    
    params_default = {
        'a4' : 1.0,
        'a2' : 6.0,
        'a1' : 1.0,
        'k' : 1.0,
        'dim' : 2
    }
    
    def __init__(self, params=None):
        # set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params

        # useful variables
        self.dim = self.params['dim']
    
    def __call__(self, configuration):
        dimer_energy = self.params['a4'] * configuration[:, 0] ** 4\
            - self.params['a2'] * configuration[:, 0] ** 2\
            + self.params['a1'] * configuration[:, 0]
            
        oscillator_energy = 0.0
        if self.dim == 2:
            oscillator_energy = (self.params['k'] / 2.0) * configuration[:, 1] ** 2
        if self.dim > 2:
            oscillator_energy = np.sum((self.params['k'] / 2.0) * configuration[:, 1:] ** 2, axis=1)
        return  dimer_energy + oscillator_energy
    
    def energy(self, x):
        return self(x)
    
    def energy_tf(self, x):
        return self(x)


class OpenMMEnergy(object):
        
    def __init__(self, openmm_system, openmm_integrator, length_scale, n_atoms=None, openmm_integrator_args=None, n_steps=0):
        self._length_scale = length_scale
        self._openmm_integrator = openmm_integrator(*openmm_integrator_args)

        self._openmm_context = openmm.Context(openmm_system, self._openmm_integrator)
        
        kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self._unit_reciprocal = 1. / (self._openmm_integrator.getTemperature() * kB_NA)

        self.energy_tf = wrap_energy_as_tf_op(self.__call__)
        
        self.n_steps = n_steps

        self.dim = 3*n_atoms

        self.natoms = n_atoms
        self.atom_indices = np.arange(self.dim).reshape(self.natoms, 3).astype(np.int32)

    def _reduce_units(self, x):
        return x * self._unit_reciprocal
    
    def _assign_openmm_positions(self, configuration):
        positions = openmm.unit.Quantity(
            value=configuration.reshape(-1, 3), 
            unit=self._length_scale)
        self._openmm_context.setPositions(positions)
    
    def _get_energy_from_openmm_state(self, state):
        energy_quantity = state.getPotentialEnergy()
        return self._reduce_units(energy_quantity)
    
    def _get_gradient_from_openmm_state(self, state):
        forces_quantity = state.getForces(asNumpy=True)
        return -1. * np.ravel(self._reduce_units(forces_quantity) * self._length_scale)
    
    def _simulate(self, n_steps):
        self._openmm_integrator.step(n_steps)

    def _get_state(self, **kwargs):
        return self._openmm_context.getState(**kwargs)

    def __call__(self, batch, n_steps=0):
        """batch: (B, N*D) """
        
        gradients = np.zeros_like(batch, dtype=batch.dtype)
        energies = np.zeros((batch.shape[0], 1), dtype=batch.dtype)    
        
        # force `np.float64` for OpenMM
        batch_ = batch.astype(np.float64)
        
        for batch_idx, configuration in enumerate(batch_):
            if np.all(np.isfinite(configuration)):
                self._assign_openmm_positions(configuration)
                if n_steps > 0:
                    self._simulate(n_steps)
                state = self._get_state(getForces=True, getEnergy=True)
                energies[batch_idx] = self._get_energy_from_openmm_state(state)
                # zero out gradients for non-finite energies 
                if np.isfinite(energies[batch_idx]):
                    gradients[batch_idx] = self._get_gradient_from_openmm_state(state)
                    
        return energies, gradients
    
    def energy(self, batch):
        """batch: (B, N*D) """
        
        energies = np.zeros(batch.shape[0], dtype=batch.dtype)    
        
        # force `np.float64` for OpenMM
        batch_ = batch.astype(np.float64)
        
        for batch_idx, configuration in enumerate(batch_):
            if np.all(np.isfinite(configuration)):
                self._assign_openmm_positions(configuration)
                if self.n_steps > 0:
                    self._simulate(self.n_steps)
                state = self._get_state(getEnergy=True)
                energies[batch_idx] = self._get_energy_from_openmm_state(state)

        return energies

    

def wrap_energy_as_tf_op(compute_energy, n_steps=0):
    """Wraps an energy evaluator in a tensorflow op that returns gradients
        
            `compute_energy`:    Callable that takes a (B, N*D) batch of `configuration` and returns the total energy (scalar)
                                 over all batches (unaveraged) and the (B*N, D) tensor of all gradients wrt to the batch
                                 of configurations.
    """
    
    @tf.custom_gradient
    def _energy(configuration):
        """Actual tf op that is evaluated in the `tf.Graph()` built by `keras.Model.compile()`
           
               `configuration`: (B, D*N) tensor containing the B batches of D*N dimensional configurations.
            
            Returns
                        `energy`:   Scalar containg the average energy of the whole batch
                        `grad_fun`: Function returning the gradients wrt configuration given gradient wrt output  according to the chain rule
        """
        n_batch, n_system_dim = configuration.get_shape().as_list()
        dtype = configuration.dtype
        
        batch_size, ndims = configuration.shape

        # here we can call our python function using the `tf.py_func` wrapper
        # important to note: this has to be executed on the master node (only important for distributed computing)

        potential_energy, gradients = tf.py_func(func=compute_energy, inp=[configuration], Tout=[dtype, dtype])
        potential_energy.set_shape((n_batch, 1))
        gradients.set_shape((n_batch, n_system_dim))

        
        def _grad_fn(grad_out):
            """Function returing the gradeint wrt configuration given the gradient wrt output according to the chain rule:
            
                    takes `dL/df`
                    and returns `dL/dx = dL/df * df/dx`
            """
            # enforce (B, 1) for scalar outputs
            if len(grad_out.get_shape().as_list()) < 2:
                grad_out = tf.expand_dims(grad_out, axis=-1)
            gradients_in = grad_out * gradients
            return gradients_in
        return potential_energy, _grad_fn
    return _energy
