import os
import numpy as np
from scipy.io import savemat


def sample_dirichlet_grf(N=1024, K=200, L=1.0, seed=None):
    """
    Sample u ~ N(0, 49(-Delta + 7I)^(-2.5))
    on (0, L) with homogeneous Dirichlet boundary conditions.
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0, L, N + 1)
    u = np.zeros_like(x)

    # k >= 1 Dirichlet eigenfunctions: phi_k = sqrt(2/L) sin(k pi x / L)
    for k in range(1, K + 1):
        lam = (k * np.pi / L) ** 2
        coeff = np.sqrt(49 * (lam + 7) ** (-2.5))
        xi = rng.normal()
        phi_k = np.sqrt(2 / L) * np.sin(k * np.pi * x / L)
        u += xi * coeff * phi_k

    return x, u


N_samples = 100
N_grid = 1024
K = 200

u0_all = np.zeros((N_samples, N_grid + 1))

rng = np.random.default_rng(0)
for i in range(N_samples):
    _, u0_all[i] = sample_dirichlet_grf(
        N=N_grid, K=K, seed=int(rng.integers(1e9))
    )

os.makedirs('./data', exist_ok=True)
np.save('./data/data_initial_KPP.npy', u0_all)
print(f"Saved {N_samples} initial conditions to ./data/data_initial_KPP.npy")
print(f"  shape: {u0_all.shape} (samples x grid points)")
print(f"  min/max: {u0_all.min():.4f} / {u0_all.max():.4f}")
