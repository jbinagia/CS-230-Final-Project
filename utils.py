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
# Atomic potentials
#################################################################################

def lj_potential(r, sig, eps):
    r6 = (sig / r) ** 6
    r12 = r6 * r6
    return 4.0 * eps * (r12 - r6)

def harmonic_potential(r, r0, k):
    return 0.5 * k * (r - r0)**2