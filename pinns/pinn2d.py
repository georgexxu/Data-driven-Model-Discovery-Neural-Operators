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
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = activation
        self.relu_k = relu_k

    def _activate(self, z):
        if self.activation == "tanh":
            return torch.tanh(z)
        if self.activation == "relu":
            return torch.relu(z).pow(self.relu_k)
        raise ValueError(f"Unknown activation: {self.activation!r}. Use 'tanh' or 'relu'.")

    def forward(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        u = self._activate(self.fc1(xy))
        u = self.fc2(u)
        return x * (1 - x) * y * (1 - y) * u  # u = 0 on boundary of [0, 1]^2


def laplacian(model, xy):
    u = model(xy)
    grad_u = torch.autograd.grad(
        u, xy, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x, xy, torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y, xy, torch.ones_like(u_y), create_graph=True, retain_graph=True
    )[0][:, 1:2]
    return u_xx + u_yy


def pde_loss(model, xy, f):
    lap_u = laplacian(model, xy)
    return torch.mean((lap_u.squeeze() + f) ** 2)


def relative_l2_error(model, xy_test, u_test, l2_norm_exact):
    with torch.no_grad():
        u_pred = model(xy_test).squeeze()
        return torch.norm(u_pred - u_test) / l2_norm_exact


def make_interior_grid(n_per_axis, device):
    x_1d = torch.linspace(0, 1, n_per_axis + 2)[1:-1]
    y_1d = torch.linspace(0, 1, n_per_axis + 2)[1:-1]
    X, Y = torch.meshgrid(x_1d, y_1d, indexing="ij")
    xc = X.reshape(-1).to(device)
    yc = Y.reshape(-1).to(device)
    xy = torch.stack([xc, yc], dim=1).clone().requires_grad_(True)
    return xy, xc, yc, X, Y


def exact_solution(xc, yc, omega):
    return torch.sin(omega * xc) * torch.sin(omega * yc) / 100.0


def source_term(xc, yc, omega):
    # For u = sin(omega x) sin(omega y) / 100: Delta u = -2 omega^2 sin(omega x) sin(omega y) / 100
    # PDE: Delta u + f = 0  =>  f = 2 omega^2 sin(omega x) sin(omega y) / 100
    return 2 * omega**2 * torch.sin(omega * xc) * torch.sin(omega * yc) / 100.0


# PDE: Delta u + f(x, y) = 0 on (0, 1)^2, u = 0 on boundary (Dirichlet)
omega = 10.0
hidden_size = 100
activation = "tanh"  # "tanh" or "relu"
relu_k = 2           # power k for ReLU^k (only used when activation="relu")
num_epochs = 6000 # 6000
print_interval = 1000
lbfgs_steps = 100
lbfgs_max_iter = 20

n_interior = 50
xy, xc, yc, _, _ = make_interior_grid(n_interior, device)
f = source_term(xc, yc, omega)
u_exact = exact_solution(xc, yc, omega)

n_test = 100
xy_test, xc_test, yc_test, X_test, Y_test = make_interior_grid(n_test, device)
xy_test = xy_test.detach()
u_test = exact_solution(xc_test, yc_test, omega)
l2_norm_exact = torch.norm(u_test)

model = PINN(hidden_size, activation=activation, relu_k=relu_k).to(device)
adam_optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_history = []
l2_history = []

print(f"Activation: {activation}" + (f" (k={relu_k})" if activation == "relu" else ""))
print("=== Adam ===")
t0 = time.time()
for epoch in range(1, num_epochs + 1):
    loss = pde_loss(model, xy, f)

    adam_optimizer.zero_grad()
    loss.backward()
    adam_optimizer.step()

    rel_l2 = relative_l2_error(model, xy_test, u_test, l2_norm_exact)
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
        loss = pde_loss(model, xy, f)
        loss.backward()
        return loss

    loss = lbfgs_optimizer.step(closure)
    rel_l2 = relative_l2_error(model, xy_test, u_test, l2_norm_exact)
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

with torch.no_grad():
    u_pred = model(xy_test).squeeze().cpu()
    u_exact_cpu = u_test.cpu()
    X_cpu = X_test.cpu()
    Y_cpu = Y_test.cpu()
    u_exact_2d = u_exact_cpu.reshape(X_cpu.shape)
    u_pred_2d = u_pred.reshape(X_cpu.shape)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
im0 = axes[0].contourf(X_cpu, Y_cpu, u_exact_2d, levels=50)
axes[0].set_title("Exact")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].contourf(X_cpu, Y_cpu, u_pred_2d, levels=50)
axes[1].set_title("PINN")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].contourf(X_cpu, Y_cpu, u_pred_2d - u_exact_2d, levels=50)
axes[2].set_title("Error")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
plt.colorbar(im2, ax=axes[2])

fig.tight_layout()
plt.show()

plt.figure()
plt.plot(l2_history)
plt.yscale("log")
plt.xlabel("epoch / L-BFGS step")
plt.ylabel("relative L2 error")
plt.title("Training error (Adam + L-BFGS)")
plt.show()
