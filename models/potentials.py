# Potentials used for interatomic interactions

# Distance potentials

def lj_potential(r, sig, eps):
    r6 = (sig / r) ** 6
    r12 = r6 * r6
    return 4.0 * eps * (r12 - r6)

def harmonic_potential(r, r0, k):
    return 0.5 * k * (r - r0)**2

# Angle potentials?

# Dihedral potentials?

# Use OpenMM wrapper for more complex potentials?