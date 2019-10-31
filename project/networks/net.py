"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# defining RealNVP network (https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb)
class RealNVP(nn.Module): # base class Module
    def __init__(self, nets, nett, mask, prior, input_dimension):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))]) # translation function (net)
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))]) # scaling function (net)
        # nn.ModuleList is basically just like a Python list, used to store a desired number of nn.Module’s.
        self.logp = 1.0 # initialize to 1
        self.orig_dimension = input_dimension # tuple describing original dim. of system. e.g. Ising Model with N = 8 would be (8,8)

    def g(self, z):
        x = z
        for i in range(len(self.t)): # for each layer
            x_ = x*self.mask[i] # splitting features between channels.
                                # features selected here used to compute s(x) and f(x) but not updated themselves yet.
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        # new_zeros(size) returns a Tensor of size "size" filled with 0s
        for i in reversed(range(len(self.t))): # move backwards through layers
            z_ = self.mask[i] * z # tensor of size num samples x num features
            s = self.s[i](z_) * (1-self.mask[i]) # self.s[i] is the entire sequence of scaling operations
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
            # each pass through here applies all operations defined in nets() and all the ones defined in nett()
        # self.s[1](z_) is not the same as self.s[3](z_)
        return z, log_det_J

    def forward(self, x):
        z, self.logp = self.f(x)
        return z

    def log_prob(self,x):
        z, logp = self.f(x) # z = f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x

    def my_loss(self, batch, w_ml = 1.0, w_kl = 0.0, w_rc = 0.0):
        return w_ml*self.loss_ml(batch) + w_kl*self.loss_kl(batch) + w_rc*self.loss_rc(batch)

    def loss_ml(self, batch):
        boltzmann_weights = calculate_weights(batch)
        z, log_det_J = self.f(batch)
        return expected_value(0.5*(torch.norm(z,dim=1) - log_det_J), boltzmann_weights)

    def loss_kl(self, z):
        return 0.0

    def loss_rc(self, batch):
        return 0.0

def calculate_weights(batch):
    weights = batch.new_ones(batch.shape[0])
    return weights

def expected_value(observable, weights):
    return torch.dot(observable,weights)

def realnvp_loss_fn(z, model):
    """
    """

    return -(model.prior.log_prob(z) + model.logp).mean()

# Define performance metrics related to our network architecture
def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
