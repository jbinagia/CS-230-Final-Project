import math
import numpy as np

#################################################################################
# Transformations
#################################################################################

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

#################################################################################

def separation_array(reference, configuration, box = None):
    """
    Calculate all possible separation vectors between a reference set and another
    configuration.

    If there are ``n`` positions in `reference` and ``m`` positions in
    `configuration`, a separation array of shape ``(n, m, d)`` will be computed,
    where ``d`` is the dimensionality of each vector.

    If the optional argument `box` is supplied, the minimum image convention is
    applied when calculating separations.
    """    
    refdim =  reference.shape[-1]
    confdim = configuration.shape[-1]
    if refdim != confdim:
        raise ValueError("Configuration dimension of {0} not equal to "
            "reference dimension of {1}".format(confdim, refdim))

    # Do the whole thing by broadcasting
    separations = reference[:, np.newaxis] - configuration
    if box is not None:
        box.min_image(separations)
    return separations

def distance_array(reference, configuration, box = None):
    """
    Like above, but with the L2 norm distances.
    """    
    seps = separation_array(reference, configuration, box = box)
    return np.linalg.norm(seps, axis = len(seps.shape) - 1)

#################################################################################
# Atomic potentials
#################################################################################

def lj_potential(r, sig, eps):
    r6 = (sig / r) ** 6
    r12 = r6 * r6
    return 4.0 * eps * (r12 - r6)

def harmonic_potential(r, r0, k):
    return 0.5 * k * (r - r0)**2