import os
import numpy as np

def sample_neumann_grf(N=1024, K=200, L=2 * np.pi, seed=None):
    """
    Sample u ~ N(0, 49(-Delta + 7I)^(-2.5))
    on (0, L) with pure Neumann boundary conditions.
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0, L, N + 1)
    u = np.zeros_like(x)

    # k = 0 Neumann eigenfunction: constant mode
    xi0 = rng.normal()
    lam0 = 0.0
    coeff0 = np.sqrt(49 * (lam0 + 7) ** (-2.5))
    u += xi0 * coeff0 * np.ones_like(x) / np.sqrt(L)

    # k >= 1 Neumann eigenfunctions
    for k in range(1, K + 1):
        lam = (k * np.pi / L) ** 2
        coeff = np.sqrt(49 * (lam + 7) ** (-2.5))
        xi = rng.normal()

        phi_k = np.sqrt(2 / L) * np.cos(k * np.pi * x / L)
        u += xi * coeff * phi_k

    return x, u


N_samples = 50
N_grid = 1024
K = 200

A_all = np.zeros((N_samples, N_grid + 1))
S_all = np.zeros((N_samples, N_grid + 1))

rng = np.random.default_rng(0)
for i in range(N_samples):
    _, A_all[i] = sample_neumann_grf(N=N_grid, K=K, seed=int(rng.integers(1e9)))
    _, S_all[i] = sample_neumann_grf(N=N_grid, K=K, seed=int(rng.integers(1e9)))

os.makedirs('./data', exist_ok=True)
np.save('./data/data_initial_GS.npy', np.stack([A_all, S_all], axis=0))
print(f"Saved {N_samples} initial conditions to ./data/data_initial_GS.npy")
print(f"  shape: {A_all.shape} (samples x grid points) for A and S")
