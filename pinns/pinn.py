import sys
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self, hidden_size, activation="tanh", relu_k=1):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = activation
        self.relu_k = relu_k

    def _activate(self, z):
        if self.activation == "tanh":
            return torch.tanh(z)
        if self.activation == "relu":
            return torch.relu(z).pow(self.relu_k)
        raise ValueError(f"Unknown activation: {self.activation!r}. Use 'tanh' or 'relu_k'.")

    def forward(self, x):
        u = self._activate(self.fc1(x))
        u = self.fc2(u)
        return x * (1 - x) * u  # hard constraint on the boundary conditions


def pde_loss(model, x, b):
    u = model(x)
    du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]
    return torch.mean((d2u.squeeze() + b) ** 2)


def relative_l2_error(model, x_test, u_test, l2_norm_exact):
    with torch.no_grad():
        u_pred = model(x_test.view(-1, 1)).squeeze()
        return torch.norm(u_pred - u_test) / l2_norm_exact


# PDE: u'' + sin(omega x) = 0 on (0, 1), with u(0) = u(1) = 0
omega = 10.0
hidden_size = 100
activation = "tanh"  # "tanh" or "relu_k"
relu_k = 2           # power k for ReLU^k activation (only used when activation="relu_k")
num_epochs = 6000
print_interval = 1000
lbfgs_steps = 100
lbfgs_max_iter = 20

n_interior = 500
xc = torch.linspace(0, 1, n_interior + 2)[1:-1].to(device)
x = xc.view(-1, 1).clone().requires_grad_(True)

b = torch.sin(omega * xc)
u_exact = (
    torch.sin(omega * xc) / 100.0
    - xc * torch.sin(torch.tensor(omega)) / 100.0
)

n_test = 1002
x_test = torch.linspace(0, 1, n_test)[1:-1].to(device)
u_test = (
    torch.sin(omega * x_test) / 100.0
    - x_test * torch.sin(torch.tensor(omega)) / 100.0
)
l2_norm_exact = torch.norm(u_test)

model = PINN(hidden_size, activation=activation, relu_k=relu_k).to(device)
adam_optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_history = []
l2_history = []

print(f"Activation: {activation}" + (f" (k={relu_k})" if activation == "relu_k" else ""))
print("=== Adam ===")
t0 = time.time()
for epoch in range(1, num_epochs + 1):
    loss = pde_loss(model, x, b)

    adam_optimizer.zero_grad()
    loss.backward()
    adam_optimizer.step()

    rel_l2 = relative_l2_error(model, x_test, u_test, l2_norm_exact)
    loss_history.append(loss.item())
    l2_history.append(rel_l2.item())

    if epoch % print_interval == 0 or epoch == 1 or epoch == num_epochs:
        print(f"Epoch {epoch:6d} | loss={loss.item():.4e} | rel L2={rel_l2.item():.4e}")

adam_time = time.time() - t0
adam_best_l2 = min(l2_history)
adam_final_loss = loss_history[-1]
adam_final_l2 = l2_history[-1]
print(
    f"Adam time: {adam_time:.2f}s | final loss: {adam_final_loss:.4e} "
    f"| final rel L2: {adam_final_l2:.4e} | best rel L2: {adam_best_l2:.4e}"
)

print("\n=== L-BFGS (after Adam) ===")
lbfgs_optimizer = optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=lbfgs_max_iter,
    line_search_fn="strong_wolfe",
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
)

t0 = time.time()
for step in range(1, lbfgs_steps + 1):
    def closure():
        lbfgs_optimizer.zero_grad()
        loss = pde_loss(model, x, b)
        loss.backward()
        return loss

    loss = lbfgs_optimizer.step(closure)
    rel_l2 = relative_l2_error(model, x_test, u_test, l2_norm_exact)
    loss_history.append(loss.item())
    l2_history.append(rel_l2.item())

    if step % max(1, lbfgs_steps // 10) == 0 or step == 1 or step == lbfgs_steps:
        print(f"Step {step:4d} | loss={loss.item():.4e} | rel L2={rel_l2.item():.4e}")

lbfgs_time = time.time() - t0
lbfgs_best_l2 = min(l2_history[len(l2_history) - lbfgs_steps:])
lbfgs_final_loss = loss_history[-1]
lbfgs_final_l2 = l2_history[-1]
print(
    f"L-BFGS time: {lbfgs_time:.2f}s | final loss: {lbfgs_final_loss:.4e} "
    f"| final rel L2: {lbfgs_final_l2:.4e} | best rel L2: {lbfgs_best_l2:.4e}"
)
print(
    f"\nOverall best rel L2: {min(l2_history):.4e} "
    f"(Adam final: {adam_final_l2:.4e}, Adam best: {adam_best_l2:.4e}, "
    f"L-BFGS final: {lbfgs_final_l2:.4e}, L-BFGS best: {lbfgs_best_l2:.4e})"
)

u_pred = model(x.view(-1, 1)).squeeze().detach().cpu()
x_cpu = xc.cpu()
u_exact_cpu = u_exact.cpu()

plt.figure()
plt.plot(x_cpu, u_exact_cpu, label="Exact")
plt.plot(x_cpu, u_pred, label="PINN")
plt.legend()
plt.title("Solution")
plt.show()

plt.figure()
plt.plot(l2_history)
plt.yscale("log")
plt.xlabel("epoch / L-BFGS step")
plt.ylabel("relative L2 error")
plt.title("Training error (Adam + L-BFGS)")
plt.show()
