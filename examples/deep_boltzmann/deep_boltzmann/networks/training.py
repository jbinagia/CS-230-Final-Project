import sys, numbers
import numpy as np
import tensorflow as tf
import keras
from deep_boltzmann.networks.invertible_coordinate_transforms import MixedCoordinatesTransformation

class MLTrainer(object):
    def __init__(self, bg, optimizer=None, lr=0.001, clipnorm=None,
                 std=1.0, reg_Jxz=0.0, save_test_energies=False):
        self.bg = bg
        self.save_test_energies = save_test_energies

        if optimizer is None:
            if clipnorm is None:
                optimizer = keras.optimizers.adam(lr=lr)
            else:
                optimizer = keras.optimizers.adam(lr=lr, clipnorm=clipnorm)

        def loss_ML_normal(y_true, y_pred):
            return -bg.log_likelihood_z_normal(std=std)
        def loss_ML_lognormal(y_true, y_pred):
            return -bg.log_likelihood_z_lognormal(std=std)
        def loss_ML_cauchy(y_true, y_pred):
            return -bg.log_likelihood_z_cauchy(scale=std)
        def loss_ML_normal_reg(y_true, y_pred):
            return -bg.log_likelihood_z_normal(std=std) + reg_Jxz*bg.reg_Jxz_uniform()
        def loss_ML_lognormal_reg(y_true, y_pred):
            return -bg.log_likelihood_z_lognormal(std=std) + reg_Jxz*bg.reg_Jxz_uniform()
        def loss_ML_cauchy_reg(y_true, y_pred):
            return -bg.log_likelihood_z_cauchy(scale=std) + reg_Jxz*bg.reg_Jxz_uniform()

        print('Compiling the graph')
        if bg.prior == 'normal':
            if reg_Jxz == 0:
                bg.Txz.compile(optimizer, loss=loss_ML_normal)
            else:
                bg.Txz.compile(optimizer, loss=loss_ML_normal_reg)
        elif bg.prior == 'lognormal':
            if reg_Jxz == 0:
                bg.Txz.compile(optimizer, loss=loss_ML_lognormal)
            else:
                bg.Txz.compile(optimizer, loss=loss_ML_lognormal_reg)
        elif bg.prior == 'cauchy':
            if reg_Jxz == 0:
                bg.Txz.compile(optimizer, loss=loss_ML_cauchy)
            else:
                bg.Txz.compile(optimizer, loss=loss_ML_cauchy_reg)
        else:
            raise NotImplementedError('ML for prior ' + bg.prior + ' is not implemented.')

        self.loss_train = []
        self.loss_val = []
        if save_test_energies:
            self.energies_x_val = []
            self.energies_z_val = []


    def train(self, x_train, x_val=None, u_prior=None,
              epochs=2000, batch_size=1024, verbose=1, save_test_energies=False):
        """
        Parameters
        ----------
        u_prior : Prior energy for every sample. Learn the difference enerry

        """
        N = x_train.shape[0]
        I = np.arange(N)
        y = np.zeros((batch_size, self.bg.dim))
        for e in range(epochs):
            # sample batch
            x_batch = x_train[np.random.choice(I, size=batch_size, replace=True)]
            l = self.bg.Txz.train_on_batch(x=x_batch, y=y)
            self.loss_train.append(l)

            # validate
            if x_val is not None:
                xval_batch = x_val[np.random.choice(I, size=batch_size, replace=True)]
                l = self.bg.Txz.test_on_batch(x=xval_batch, y=y)
                self.loss_val.append(l)
                if self.save_test_energies:
                    z = self.bg.sample_z(nsample=batch_size)
                    xout = self.bg.transform_zx(z)
                    self.energies_x_val.append(self.bg.energy_model.energy(xout))
                    zout = self.bg.transform_xz(xval_batch)
                    self.energies_z_val.append(self.bg.energy_z(zout))

            # print
            if verbose > 0:
                str_ = 'Epoch ' + str(e) + '/' + str(epochs) + ' '
                str_ += self.bg.Txz.metrics_names[0] + ' '
                str_ += '{:.4f}'.format(self.loss_train[-1]) + ' '
                if x_val is not None:
                    str_ += '{:.4f}'.format(self.loss_val[-1]) + ' '
                print(str_)
                sys.stdout.flush()


class FlexibleTrainer(object):
    def __init__(self, bg, optimizer=None, lr=0.001, batch_size=1024,
                 high_energy=100, max_energy=1e10, std=1.0, temperature=1.0,
                 w_KL=1.0, w_ML=1.0, w_RC=0.0, w_L2_angle=0.0,
                 rc_func=None, rc_min=0.0, rc_max=1.0,
                 weigh_ML=True, mapper=None):
        """
        Parameters:
        -----------
        """
        self.bg = bg
        self.lr = lr
        self.high_energy = high_energy
        self.max_energy = max_energy
        self.std = std
        self.weighML = weigh_ML
        self.mapper = mapper
        self.rc_func = rc_func

        # Temperature and batch size
        import numbers
        if isinstance(temperature, numbers.Number):
            temperature = np.array([temperature])
        else:
            temperature = np.array(temperature)
        self.temperature = temperature.astype(np.float32)
        # redefine batch size to be a multiple of temperatures
        batch_size_per_temp = int(batch_size / self.temperature.size)
        self.batch_size = int(self.temperature.size * batch_size_per_temp)
        tidx = np.tile(np.arange(self.temperature.size), batch_size_per_temp)
        self.tfac = self.temperature[tidx]

        inputs = [bg.input_x, bg.input_z]
        outputs = [bg.output_z, bg.output_x]
        if weigh_ML:
            losses = [self.loss_ML_weighted, self.loss_KL]
        else:
            losses = [self.loss_ML, self.loss_KL]
        loss_weights = [w_ML, w_KL]
        if w_RC > 0.0:
            self.gmeans = np.linspace(rc_min, rc_max, 11)
            self.gstd = (rc_max - rc_min) / 11.0
            outputs.append(bg.output_x)
            losses.append(self.loss_RC)
            loss_weights.append(w_RC)
        if w_L2_angle > 0.0:
            outputs.append(bg.output_x)
            losses.append(self.loss_L2_angle_penalization)
            loss_weights.append(w_L2_angle)

        # build estimator
        if optimizer is None:
            optimizer = keras.optimizers.adam(lr=lr)

        # assemble model
        self.dual_model = keras.models.Model(inputs=inputs, outputs=outputs)
        self.dual_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

        # training loop
        dummy_output = np.zeros((self.batch_size, bg.dim))
        self.y = [dummy_output for o in outputs]

        self.loss_train = []
        self.acceptance_rate = []

    def loss_ML(self, y_true, y_pred):
        z = self.bg.output_z
        Jxz = self.bg.log_det_Jxz[:, 0]
        LL = Jxz - (0.5 / (self.std**2)) * tf.reduce_sum(z**2, axis=1)
        return -LL

    def loss_ML_weighted(self, y_true, y_pred):
        from deep_boltzmann.util import linlogcut
        x = self.bg.input_x
        z = self.bg.output_z
        Jxz = self.bg.log_det_Jxz[:, 0]
        LL = Jxz - (0.5 / (self.std**2)) * tf.reduce_sum(z**2, axis=1)
        # compute energies
        E = self.bg.energy_model.energy_tf(x) / self.tfac
        Ereg = linlogcut(E, self.high_energy, self.max_energy, tf=True)
        # weights
        Ez = tf.reduce_sum(z**2, axis=1)/(2.0*self.tfac)
        logW = -Ereg + Ez - Jxz
        logW = logW - tf.reduce_max(logW)
        weights = tf.exp(logW)
        # weighted ML
        weighted_negLL = -self.batch_size * (weights * LL) / tf.reduce_sum(weights)
        return weighted_negLL

    def loss_KL(self, y_true, y_pred):
        """ Computes the KL divergence with respect to z|x and the Boltzmann distribution
        """
        from deep_boltzmann.util import linlogcut, _clip_high_tf, _linlogcut_tf_constantclip
        x = self.bg.output_x
        # compute energy
        E = self.bg.energy_model.energy_tf(x)
        # regularize using log
        Ereg = linlogcut(E, self.high_energy, self.max_energy, tf=True)
        #Ereg = _linlogcut_tf_constantclip(E, high_energy, max_energy)
        # gradient_clip(bg1.energy_model.energy_tf, 1e16, 1e20)
        free_energy = Ereg / self.tfac - self.bg.log_det_Jzx[:, 0]
        #free_energy /=
        return free_energy

    def loss_RC(self, y_true, y_pred):
        return -self.bg.rc_entropy(self.rc_func, self.gmeans, self.gstd)

    def loss_L2_angle_penalization(self, y_true, y_pred):
        losses = []
        for layer in self.bg.layers:
            if hasattr(layer, "angle_loss"):
                losses.append(layer.angle_loss)
        loss = sum(losses)
        return loss

    def train(self, x_train, epochs=2000, verbose=1):
        I = np.arange(x_train.shape[0])

        for e in range(epochs):
            # sample batch
            Isel = np.random.choice(I, size=self.batch_size, replace=True)
            x_batch = x_train[Isel]
            w_batch = np.sqrt(self.tfac)[:, None] * np.random.randn(self.batch_size, self.bg.dim)
            l = self.dual_model.train_on_batch(x=[x_batch, w_batch], y=self.y)
            self.loss_train.append(l)

            # print
            if verbose > 0:
                str_ = 'Epoch ' + str(e) + '/' + str(epochs) + ' '
                for i in range(len(self.dual_model.metrics_names)):
                    str_ += self.dual_model.metrics_names[i] + ' '
                    str_ += '{:.4f}'.format(self.loss_train[-1][i]) + ' '
                print(str_)
                sys.stdout.flush()

class ForceMatching(object):
    def __init__(self, bg, optimizer=None, lr=0.001, batch_size=1024, clipnorm=None,
                 high_energy=100, max_energy=1e10, temperature=1.0,
                 w_force=1.0, w_energy=1.0,
                 mapper=None):
        """
        Parameters:
        -----------
        """
        self.bg = bg
        self.lr = lr
        self.high_energy = high_energy
        self.max_energy = max_energy
        self.w_force = w_force
        self.w_energy = w_energy
        self.mapper = mapper

        # Temperature and batch size
        import numbers
        if isinstance(temperature, numbers.Number):
            temperature = np.array([temperature])
        else:
            temperature = np.array(temperature)
        self.temperature = temperature.astype(np.float32)
        # redefine batch size to be a multiple of temperatures
        batch_size_per_temp = int(batch_size / self.temperature.size)
        self.batch_size = int(self.temperature.size * batch_size_per_temp)
        tidx = np.tile(np.arange(self.temperature.size), batch_size_per_temp)
        self.tfac = self.temperature[tidx]

        # build estimator
        if optimizer is None:
            if clipnorm is None:
                self.optimizer = keras.optimizers.adam(lr=lr)
            else:
                self.optimizer = keras.optimizers.adam(lr=lr, clipnorm=clipnorm)

        # assemble model
        self.Fxz = keras.models.Model(inputs=bg.input_x, outputs=bg.output_z)
        self.Fxz.compile(optimizer=self.optimizer, loss=self.loss_force_matching)
        self.Fzx = keras.models.Model(inputs=bg.input_z, outputs=bg.output_x)

        # loss history
        self.loss_train = []
        self.gradient_norm = []

    def loss_force_matching(self, y_true, y_pred):
        from deep_boltzmann.util import linlogcut

        x_in = self.bg.input_x
        z_out = self.bg.output_z
        Jxz = self.bg.log_det_Jxz[:, 0]
        enerx = self.bg.energy_model.energy_tf(x_in)
        enerx = linlogcut(enerx, self.high_energy, self.max_energy, tf=True)

        neg_log_px = 0.5 * tf.reduce_sum(z_out**2, axis=1) - Jxz

        fx = -tf.gradients(neg_log_px, x_in)[0]
        fx_ref = -tf.gradients(enerx, x_in)[0]

        force_error = fx - fx_ref
        energy_error = neg_log_px - enerx
        mse = tf.reduce_mean(self.w_force * force_error**2, axis=1) + self.w_energy * energy_error**2
        return mse

    def train(self, x_train=None, x_batch_size=None, epochs=2000, verbose=1):
        if x_train is None:
            y_dummy = np.zeros((self.batch_size, self.bg.dim))
        else:
            if x_batch_size is None:
                x_batch_size = self.batch_size
            y_dummy = np.zeros((self.batch_size + x_batch_size, self.bg.dim))


        for e in range(epochs):
            # predict sample
            z = np.sqrt(self.tfac)[:, None] * np.random.randn(self.batch_size, self.bg.dim)
            x_in = self.Fzx.predict(z)
            if self.mapper is not None:
                x_in = self.mapper.map(x_in)


            if x_train is not None:
                I = np.random.choice(x_train.shape[0], size=x_batch_size)
                x_in = np.vstack([x_in, x_train[I]])

            # now train with this sample
            l = self.Fxz.train_on_batch(x=x_in, y=y_dummy)
            self.loss_train.append(l)
            #norm = np.sqrt(sum(np.sum(keras.backend.get_value(w)) for w in self.optimizer.weights))
            #self.gradient_norm.append(norm)

            # print
            if verbose > 0:
                str_ = 'Epoch ' + str(e) + '/' + str(epochs) + ' '
                str_ += '{:.4f}'.format(self.loss_train[-1]) + ' '
                print(str_)
                sys.stdout.flush()


class ParticleFilterDerived(FlexibleTrainer):
    def __init__(self, bg, X0, capacity, optimizer=None, lr=0.001, batch_size=1024,
                 high_energy=100, max_energy=1e10, std=1.0, w_KL=1.0, w_ML=1.0, w_RC=0.0,
                 rc_func=None, rc_min=0.0, rc_max=1.0,
                 weigh_ML=True, mapper=None):
        """
        Parameters:
        -----------
        X0 : array or None
            If none, the Boltzmann Generator will be used to generate samples to fill the buffer.
            If given, the buffer will be filled with random samples from X0.
        """
        super().__init__(bg, optimizer=None, lr=lr, batch_size=batch_size,
                         high_energy=high_energy, max_energy=max_energy, std=std, w_KL=w_KL, w_ML=w_ML, w_RC=w_RC,
                         rc_func=rc_func, rc_min=rc_min, rc_max=rc_max,
                         weigh_ML=weigh_ML, mapper=mapper)
        # initial data processing
        self.I = np.arange(capacity)
        if X0 is None:
            _, self.X, _, _, _ = bg.sample(temperature=self.temperature, nsample=capacity)
        else:
            I_X0 = np.arange(X0.shape[0])
            Isel = np.random.choice(I_X0, size=capacity, replace=True)
            self.X = X0[Isel]

    def train(self, epochs=2000, stepsize=1.0, verbose=1):
        for e in range(epochs):
            # sample batch
            Isel = np.random.choice(self.I, size=self.batch_size, replace=True)
            x_batch = self.X[Isel]
            w_batch = np.sqrt(self.temperature) * np.random.randn(self.batch_size, self.bg.dim)
            l = self.dual_model.train_on_batch(x=[x_batch, w_batch], y=self.y)
            self.loss_train.append(l)

            # Do an MCMC step with the current BG

            # First recompute Z and logW
            z_batch, Jxz_batch = self.bg.transform_xzJ(x_batch)
            logW_old = self.bg.energy_model.energy(x_batch) / self.temperature + Jxz_batch

            # New step
            z_batch_new = z_batch + stepsize * np.sqrt(self.temperature) * np.random.randn(z_batch.shape[0], z_batch.shape[1])
            x_batch_new, Jzx_batch_new = self.bg.transform_zxJ(z_batch_new)
            logW_new = self.bg.energy_model.energy(x_batch_new) / self.temperature - Jzx_batch_new

            # Accept or reject according to target density
            rand = -np.log(np.random.rand(self.batch_size))
            Iacc = rand >= logW_new - logW_old

            # map accepted
            x_acc = x_batch_new[Iacc]
            if self.mapper is not None:
                x_acc = self.mapper.map(x_acc)
            self.X[Isel[Iacc]] = x_acc

            # acceptance rate
            pacc = float(np.count_nonzero(Iacc)) / float(self.batch_size)
            self.acceptance_rate.append(pacc)

            # print
            if verbose > 0:
                str_ = 'Epoch ' + str(e) + '/' + str(epochs) + ' '
                for i in range(len(self.dual_model.metrics_names)):
                    str_ += self.dual_model.metrics_names[i] + ' '
                    str_ += '{:.4f}'.format(self.loss_train[-1][i]) + ' '
                str_ += 'p_acc ' + str(pacc)
                print(str_)
                sys.stdout.flush()

class ResidualTrainer(object):
    def __init__(self, bg, optimizer=None, lr=0.001, batch_size=1024,
                 high_energy=100, max_energy=1e10, std=1.0, w_KL=1.0, w_RC=0.0,
                 rc_func=None, rc_min=0.0, rc_max=1.0,
                 mapper=None):
        """
        Parameters:
        -----------
        """
        self.bg = bg
        self.lr = lr
        self.batch_size = batch_size
        self.high_energy = high_energy
        self.max_energy = max_energy
        self.std = std
        self.temperature = 1.0
        self.mapper = mapper
        self.rc_func = rc_func

        self.input_x0 = keras.layers.Input((bg.dim,))
        inputs = [self.input_x0, bg.input_z]
        self.output_xtot = keras.layers.Add()([self.input_x0, bg.output_x])
        outputs = [self.output_xtot]
        losses = [self.loss_KL]
        loss_weights = [w_KL]
        if w_RC > 0.0:
            self.gmeans = np.linspace(rc_min, rc_max, 11)
            self.gstd = (rc_max - rc_min) / 11.0
            outputs.append(bg.output_x)
            losses.append(self.loss_RC)
            loss_weights.append(w_RC)

        # build estimator
        if optimizer is None:
            optimizer = keras.optimizers.adam(lr=lr)

        # assemble model
        self.dual_model = keras.models.Model(inputs=inputs, outputs=outputs)
        self.dual_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

        # training loop
        dummy_output = np.zeros((batch_size, bg.dim))
        self.y = [dummy_output for o in outputs]

        self.loss_train = []
        self.acceptance_rate = []

    def loss_KL(self, y_true, y_pred):
        from deep_boltzmann.util import linlogcut, _clip_high_tf, _linlogcut_tf_constantclip
        x = self.output_xtot
        # compute energy
        E = self.bg.energy_model.energy_tf(x) / self.temperature
        # regularize using log
        Ereg = linlogcut(E, self.high_energy, self.max_energy, tf=True)
        #Ereg = _linlogcut_tf_constantclip(E, high_energy, max_energy)
        # gradient_clip(bg1.energy_model.energy_tf, 1e16, 1e20)
        #return self.log_det_Jzx + Ereg
        explore = 1.0
        return -explore * self.bg.log_det_Jzx[:, 0] + Ereg

    def loss_RC(self, y_true, y_pred):
        return -self.bg.rc_entropy(self.rc_func, self.gmeans, self.gstd)

    def train(self, x0, epochs=2000, verbose=1):
        I = np.arange(x0.shape[0])

        for e in range(epochs):
            # sample batch
            Isel = np.random.choice(I, size=self.batch_size, replace=True)
            x_batch = x0[Isel]
            w_batch = np.sqrt(self.temperature) * np.random.randn(self.batch_size, self.bg.dim)
            l = self.dual_model.train_on_batch(x=[x_batch, w_batch], y=self.y)
            self.loss_train.append(l)

            # print
            if verbose > 0:
                str_ = 'Epoch ' + str(e) + '/' + str(epochs) + ' '
                if isinstance(l, numbers.Number):
                    str_ += self.dual_model.metrics_names[0] + ' '
                    str_ += '{:.4f}'.format(l) + ' '
                else:
                    for i in range(len(self.dual_model.metrics_names)):
                        str_ += self.dual_model.metrics_names[i] + ' '
                        str_ += '{:.4f}'.format(l[i]) + ' '
                print(str_)
                sys.stdout.flush()



class ParticleFilter(object):
    def __init__(self, bg, X0, capacity, optimizer=None, lr=0.001, batch_size=1024, 
                 high_energy=100, max_energy=1e10, std=1.0, w_KL=1.0, w_ML=1.0, w_RC=0.0, 
                 rc_func=None, rc_min=0.0, rc_max=1.0,
                 weigh_ML=True, mapper=None):
        """
        Parameters:
        -----------
        X0 : array or None
            If none, the Boltzmann Generator will be used to generate samples to fill the buffer. 
            If given, the buffer will be filled with random samples from X0.
        """
        self.bg = bg
        self.lr = lr
        self.batch_size = batch_size
        self.high_energy = high_energy
        self.max_energy = max_energy
        self.std = std
        self.temperature = 1.0
        self.weighML = weigh_ML
        self.mapper = mapper
        self.rc_func = rc_func

        inputs = [bg.input_x, bg.input_z]
        outputs = [bg.output_z, bg.output_x]        
        if weigh_ML:
            losses = [self.loss_ML_weighted, self.loss_KL]
        else:
            losses = [self.loss_ML, self.loss_KL]
        loss_weights = [w_ML, w_KL]
        if w_RC > 0.0:
            self.gmeans = np.linspace(rc_min, rc_max, 11)
            self.gstd = (rc_max - rc_min) / 11.0
            outputs.append(bg.output_x)
            losses.append(self.loss_RC)
            loss_weights.append(w_RC)

        # initial data processing
        self.I = np.arange(capacity)
        if X0 is None:
            _, self.X, _, _, _ = bg.sample(temperature=self.temperature, nsample=capacity)
        else:
            I_X0 = np.arange(X0.shape[0])
            Isel = np.random.choice(I_X0, size=capacity, replace=True)
            self.X = X0[Isel]

        # build estimator
        if optimizer is None:
            optimizer = keras.optimizers.adam(lr=lr)

        # assemble model
        self.dual_model = keras.models.Model(inputs=inputs, outputs=outputs)
        self.dual_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

        # training loop
        dummy_output = np.zeros((batch_size, bg.dim))
        self.y = [dummy_output for o in outputs]

        self.loss_train = []
        self.acceptance_rate = []
        self.stepsize = []
        
    def loss_ML(self, y_true, y_pred):
        z = self.bg.output_z
        Jxz = self.bg.log_det_Jxz[:, 0]
        LL = Jxz - (0.5 / (self.std**2)) * tf.reduce_sum(z**2, axis=1)
        return -LL

    def loss_ML_weighted(self, y_true, y_pred):
        from deep_boltzmann.util import linlogcut
        x = self.bg.input_x
        z = self.bg.output_z
        Jxz = self.bg.log_det_Jxz[:, 0]
        LL = Jxz - (0.5 / (self.std**2)) * tf.reduce_sum(z**2, axis=1)
        # compute energies
        E = self.bg.energy_model.energy_tf(x) / self.temperature
        Ereg = linlogcut(E, self.high_energy, self.max_energy, tf=True)
        # weights
        Ez = tf.reduce_sum(z**2, axis=1)/(2.0*self.temperature)
        logW = -Ereg + Ez - Jxz
        logW = logW - tf.reduce_max(logW)
        weights = tf.exp(logW)
        # weighted ML
        weighted_negLL = -self.batch_size * (weights * LL) / tf.reduce_sum(weights)
        return weighted_negLL

    def loss_KL(self, y_true, y_pred):
        return self.bg.log_KL_x(self.high_energy, self.max_energy, temperature_factors=self.temperature, explore=1.0)

    def loss_RC(self, y_true, y_pred):
        # TODO: this is not using the local entropy impl.
        return -self.bg.rc_entropy(self.rc_func, self.gmeans, self.gstd)

    def rc_plogp(self, rc_func, gmeans, gstd, ntemperatures=1):
        """ Computes the entropy along a 1D reaction coordinate

        Parameters
        ----------
        rc_func : function
            function to compute reaction coordinate
        gmeans : array
            mean positions of Gauss kernels along reaction coordinate
        gstd : float
            standard deviation of Gauss kernels along reaction coordinate
        """
        # evaluate rc
        rc = rc_func(self.bg.output_x)
        rc = tf.expand_dims(rc, axis=1)
        # kernelize all values
        kmat = tf.exp(-((rc - gmeans)**2) / (2*gstd*gstd))
        kmat += 1e-6
        kmat /= tf.reduce_sum(kmat, axis=1, keepdims=True)
        # distribute counts across temperatures
        batchsize_per_temperature = tf.cast(tf.shape(kmat)[0] / ntemperatures, tf.int32)
        nbins = tf.shape(gmeans)[0]
        kmatT = tf.transpose(tf.reshape(kmat, (batchsize_per_temperature, ntemperatures, nbins)), perm=(1, 0, 2))
        histogram = tf.reduce_mean(kmatT, axis=1)
        entropies = tf.reduce_sum(histogram * tf.log(histogram), axis=1)
        return tf.reduce_mean(entropies)

    def train(self, epochs=2000, stepsize=1.0, verbose=1):
        """
        Parameters
        ----------
        stepsize : float or None
            MCMC stepsize (in latent space, so 1 is a large step).
            If None, uses adaptive stepsize between 0.001 and 1 depending on the acceptance rate.
        """
        if stepsize is None:  # initialize stepsize when called for the first time
            if len(self.stepsize) == 0:
                self.stepsize.append(0.1)
        else:
            self.stepsize = [stepsize]

        for e in range(epochs):
            # sample batch
            Isel = np.random.choice(self.I, size=self.batch_size, replace=True)
            x_batch = self.X[Isel]
            w_batch = np.sqrt(self.temperature) * np.random.randn(self.batch_size, self.bg.dim)
            l = self.dual_model.train_on_batch(x=[x_batch, w_batch], y=self.y)
            self.loss_train.append(l)

            # Do an MCMC step with the current BG

            # First recompute Z and logW
            z_batch, Jxz_batch = self.bg.transform_xzJ(x_batch)
            logW_old = self.bg.energy_model.energy(x_batch) / self.temperature + Jxz_batch

            # New step
            z_batch_new = z_batch + self.stepsize[-1] * np.sqrt(self.temperature) * np.random.randn(z_batch.shape[0], z_batch.shape[1])
            x_batch_new, Jzx_batch_new = self.bg.transform_zxJ(z_batch_new)
            logW_new = self.bg.energy_model.energy(x_batch_new) / self.temperature - Jzx_batch_new

            # Accept or reject according to target density
            rand = -np.log(np.random.rand(self.batch_size))
            Iacc = rand >= logW_new - logW_old

            # map accepted
            x_acc = x_batch_new[Iacc]
            if self.mapper is not None:
                x_acc = self.mapper.map(x_acc)
            self.X[Isel[Iacc]] = x_acc

            # acceptance rate
            pacc = float(np.count_nonzero(Iacc)) / float(self.batch_size)
            self.acceptance_rate.append(pacc)

            # update stepsize
            if stepsize is None:
                if len(self.acceptance_rate) > 50:  # update stepsize
                    mean_acceptance_rate = np.mean(self.acceptance_rate[-50:])
                    if mean_acceptance_rate < 0.02:
                        self.stepsize.append(max(self.stepsize[-1] * 0.98, 0.001))
                    elif mean_acceptance_rate > 0.2:
                        self.stepsize.append(min(self.stepsize[-1] * 1.02, 1.0))
                    else:
                        self.stepsize.append(self.stepsize[-1])  # just copy old stepsize
                else:
                    self.stepsize.append(self.stepsize[-1])  # just copy old stepsize

            # print
            if verbose > 0:
                str_ = 'Epoch ' + str(e) + '/' + str(epochs) + ' '
                for i in range(len(self.dual_model.metrics_names)):
                    str_ += self.dual_model.metrics_names[i] + ' '
                    str_ += '{:.4f}'.format(self.loss_train[-1][i]) + ' '
                str_ += 'p_acc ' + str(pacc) + ' '
                if stepsize is None:
                    str_ += 'step ' + str(self.stepsize[-1])
                print(str_)
                sys.stdout.flush()
