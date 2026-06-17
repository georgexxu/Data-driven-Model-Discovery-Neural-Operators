import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# 1. Generate Darcy data
# ============================================================

def generate_permeability(n, smooth_sigma=3.0):
    """
    Generate smooth positive permeability field a(x).
    """
    z = np.random.randn(n, n)
    z = gaussian_filter(z, sigma=smooth_sigma)
    z = (z - z.mean()) / z.std()
    a = np.exp(0.5 * z)
    return a


def solve_darcy_fd(a):
    """
    Solve -div(a grad u) = 1 on [0,1]^2 with zero Dirichlet BC.
    Finite difference on n x n grid.
    """
    n = a.shape[0]
    h = 1.0 / (n - 1)

    interior = [(i, j) for i in range(1, n - 1) for j in range(1, n - 1)]
    idx = {p: k for k, p in enumerate(interior)}
    N = len(interior)

    A = lil_matrix((N, N))
    b = np.ones(N)

    for (i, j), row in idx.items():
        ae = 0.5 * (a[i, j] + a[i + 1, j])
        aw = 0.5 * (a[i, j] + a[i - 1, j])
        an = 0.5 * (a[i, j] + a[i, j + 1])
        ass = 0.5 * (a[i, j] + a[i, j - 1])

        A[row, row] = (ae + aw + an + ass) / h**2

        for neighbor, coeff in [
            ((i + 1, j), -ae / h**2),
            ((i - 1, j), -aw / h**2),
            ((i, j + 1), -an / h**2),
            ((i, j - 1), -ass / h**2),
        ]:
            if neighbor in idx:
                A[row, idx[neighbor]] = coeff

    u_inner = spsolve(A.tocsr(), b)

    u = np.zeros((n, n))
    for (i, j), k in idx.items():
        u[i, j] = u_inner[k]

    return u


def generate_dataset(num_samples, n):
    a_list = []
    u_list = []

    for _ in range(num_samples):
        a = generate_permeability(n)
        u = solve_darcy_fd(a)

        a_list.append(a)
        u_list.append(u)

    return np.array(a_list), np.array(u_list)


# ============================================================
# 2. DeepONet model
# ============================================================

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers) - 2):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim=2, width=128, p=100):
        super().__init__()

        self.branch_net = MLP([branch_dim, width, width, p])
        self.trunk_net = MLP([trunk_dim, width, width, p])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, a, y):
        """
        a: [batch, branch_dim]
        y: [batch, 2]
        """
        b = self.branch_net(a)
        t = self.trunk_net(y)
        out = torch.sum(b * t, dim=1, keepdim=True) + self.bias
        return out


# ============================================================
# 3. Prepare training data
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 32
num_train = 200
num_test = 40

print("Generating data...")
a_train, u_train = generate_dataset(num_train, n)
a_test, u_test = generate_dataset(num_test, n)

# Grid points
x = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, x, indexing="ij")
coords = np.stack([X.flatten(), Y.flatten()], axis=1)

branch_dim = n * n
num_points = n * n

# Normalize input and output
a_mean, a_std = a_train.mean(), a_train.std()
u_mean, u_std = u_train.mean(), u_train.std()

a_train_n = (a_train - a_mean) / a_std
a_test_n = (a_test - a_mean) / a_std
u_train_n = (u_train - u_mean) / u_std
u_test_n = (u_test - u_mean) / u_std

# Build pointwise DeepONet dataset
A_train = []
Y_train = []
U_train = []

for i in range(num_train):
    A_train.append(np.repeat(a_train_n[i].reshape(1, -1), num_points, axis=0))
    Y_train.append(coords)
    U_train.append(u_train_n[i].reshape(-1, 1))

A_train = np.vstack(A_train)
Y_train = np.vstack(Y_train)
U_train = np.vstack(U_train)

A_train = torch.tensor(A_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
U_train = torch.tensor(U_train, dtype=torch.float32)

dataset = TensorDataset(A_train, Y_train, U_train)
loader = DataLoader(dataset, batch_size=4096, shuffle=True)


# ============================================================
# 4. Train DeepONet
# ============================================================

model = DeepONet(branch_dim=branch_dim, width=128, p=100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 300

print("Training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for a_batch, y_batch, u_batch in loader:
        a_batch = a_batch.to(device)
        y_batch = y_batch.to(device)
        u_batch = u_batch.to(device)

        pred = model(a_batch, y_batch)
        loss = loss_fn(pred, u_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:4d}, Loss = {total_loss / len(loader):.6e}")


# ============================================================
# 5. Test
# ============================================================

def predict_solution(model, a_field):
    model.eval()

    a_input = (a_field - a_mean) / a_std
    a_input = a_input.reshape(1, -1)
    a_input = np.repeat(a_input, num_points, axis=0)

    a_tensor = torch.tensor(a_input, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(coords, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(a_tensor, y_tensor).cpu().numpy()

    pred = pred.reshape(n, n)
    pred = pred * u_std + u_mean
    return pred


errors = []

for i in range(num_test):
    pred = predict_solution(model, a_test[i])
    true = u_test[i]

    rel_error = np.linalg.norm(pred - true) / np.linalg.norm(true)
    errors.append(rel_error)

print(f"\nMean relative test error: {np.mean(errors):.4e}")
print(f"Median relative test error: {np.median(errors):.4e}")