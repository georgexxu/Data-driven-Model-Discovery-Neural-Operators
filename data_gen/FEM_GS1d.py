import os
from fenics import *
import numpy as np

set_log_level(LogLevel.ERROR)

initial_data = np.load('./data/data_initial_GS.npy')
A_all = initial_data[0]
S_all = initial_data[1]
DA = 2.5e-4
DS = 5e-4
rho = .04
mu = .065
T = .6
num_steps = 60
dt = T / num_steps
h = 1/1024
N = int(1/h)
mesh = IntervalMesh(N, 0, 2*np.pi)
cell = mesh.ufl_cell()
VA = FiniteElement("CG", cell, 1)
VS = FiniteElement("CG", cell, 1)
Ve = MixedElement((VA, VS))
V = FunctionSpace(mesh, Ve)
num_samples = A_all.shape[0] 
resultsA = np.zeros([num_samples, num_steps + 1, N + 1])
resultsS = np.zeros([num_samples, num_steps + 1, N + 1])
u_0 = Function(V)

(A, S) = TrialFunction(V)
(vA, vS) = TestFunction(V)
a = (A / dt * vA * dx + S / dt * vS * dx
     + DA * dot(grad(A), grad(vA)) * dx + DS * dot(grad(S), grad(vS)) * dx)

for i in range(num_samples):
    print(f"Sample {i + 1}/{num_samples}", flush=True)
    tt = np.zeros(2 * (N + 1))
    tt[::2] = np.flip(A_all[i, :]).astype(u_0.vector()[:].dtype)
    tt[1::2] = np.flip(S_all[i, :]).astype(u_0.vector()[:].dtype)
    u_0.vector()[:] = tt

    u_n = Function(V)
    (A_n, S_n) = split(u_n)
    u_n.assign(u_0)
    F = (A_n / dt * vA * dx + S_n / dt * vS * dx
         + (S_n * A_n**2 - (mu + rho) * A_n) * vA * dx
         + (-S_n * A_n**2 + rho * (1. - S_n)) * vS * dx)

    resultsA[i, 0, :] = np.array(np.flip(u_n.vector()[::2]))
    resultsS[i, 0, :] = np.array(np.flip(u_n.vector()[1::2]))
    for n in range(num_steps):
        u = Function(V)
        solve(a == F, u, [])
        u_n.assign(u)
        resultsA[i, n + 1, :] = np.array(np.flip(u.vector()[::2]))
        resultsS[i, n + 1, :] = np.array(np.flip(u.vector()[1::2]))
        if (n + 1) % 10 == 0 or n + 1 == num_steps:
            print(f"  step {n + 1}/{num_steps}", flush=True)

np.savez('./data/data_GS.npz', resA=resultsA, resS=resultsS)
print("Saved ./data/data_GS.npz", flush=True)
os._exit(0)  
