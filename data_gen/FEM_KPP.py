from fenics import *
import numpy as np
import os

set_log_level(LogLevel.ERROR)

# u0_all = loadmat('./data/data_Dirichlet.mat')['input']
u0_all = np.load('./data/data_initial_KPP.npy') 
# u0_all *= 100   
print(u0_all.shape) 
print(u0_all.min(), u0_all.max())  
T = 1.0
num_steps = 100
dt = T / num_steps
kappa = 1.0
lam = 1.0
h = 1/1024
N = int(1/h)
mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh, 'P', 1)
u_D = Constant(0.)
bc = [DirichletBC(V, u_D, "on_boundary")]
u_0 = Function(V)
results = np.zeros([u0_all.shape[0], num_steps + 1, N + 1])

u = TrialFunction(V)
u_n = Function(V)
v = TestFunction(V)
# Semi-implicit: implicit diffusion, explicit reaction
# (u - u_n)/dt - Laplace(u) = lam * u_n * (kappa - u_n)
f = lam * u_n * (kappa - u_n)
a = u / dt * v * dx + dot(grad(u), grad(v)) * dx
L = f * v * dx + u_n / dt * v * dx

for i in range(u0_all.shape[0]):
    print(f"Sample {i + 1}/{u0_all.shape[0]}", flush=True)
    u_0.vector()[:] = np.flip(u0_all[i, :]).astype(u_0.vector()[:].dtype)
    u_n.assign(u_0)
    results[i, 0, :] = np.array(np.flip(u_n.vector()[:]))

    for n in range(num_steps):
        u_sol = Function(V)
        solve(a == L, u_sol, bc)
        u_n.assign(u_sol)
        results[i, n + 1, :] = np.array(np.flip(u_n.vector()[:]))

np.save('./data/data_kpp.npy', results)
print("Saved ./data/data_kpp.npy", flush=True)
os._exit(0)
