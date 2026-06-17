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
    z = np.random.randn(n, n)
    z = gaussian_filter(z, sigma=smooth_sigma)
    z = (z - z.mean()) / z.std()
    a = np.exp(0.5 * z)
    return a


def solve_darcy_fd(a):
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
# 2. FNO model
# ============================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # input:  [batch, in_channels, x, y]
        # weight: [in_channels, out_channels, x, y]
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Fourier transform
        x_ft = torch.fft.rfft2(x)

        # Allocate output in Fourier space
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Low-frequency modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1,
        )

        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2,
        )

        # Inverse Fourier transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Input channels: a(x,y), x, y
        self.fc0 = nn.Linear(3, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.activation = nn.GELU()

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def get_grid(self, batchsize, size_x, size_y, device):
        x = torch.linspace(0, 1, size_x, device=device)
        y = torch.linspace(0, 1, size_y, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).repeat(batchsize, 1, 1, 1)

        return grid

    def forward(self, a):
        """
        a: [batch, n, n, 1]
        output: [batch, n, n, 1]
        """

        batchsize, size_x, size_y, _ = a.shape

        grid = self.get_grid(batchsize, size_x, size_y, a.device)

        # Concatenate input field with coordinates
        x = torch.cat((a, grid), dim=-1)

        # Lift to higher channel dimension
        x = self.fc0(x)

        # [batch, n, n, width] -> [batch, width, n, n]
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.activation(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.activation(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.activation(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = self.activation(x1 + x2)

        # [batch, width, n, n] -> [batch, n, n, width]
        x = x.permute(0, 2, 3, 1)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


# ============================================================
# 3. Prepare data
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 32
num_train = 200
num_test = 40

print("Generating data...")
a_train, u_train = generate_dataset(num_train, n)
a_test, u_test = generate_dataset(num_test, n)

# Normalize input and output
a_mean, a_std = a_train.mean(), a_train.std()
u_mean, u_std = u_train.mean(), u_train.std()

a_train_n = (a_train - a_mean) / a_std
a_test_n = (a_test - a_mean) / a_std
u_train_n = (u_train - u_mean) / u_std
u_test_n = (u_test - u_mean) / u_std

# FNO expects full fields, not pointwise samples
a_train_tensor = torch.tensor(a_train_n, dtype=torch.float32).unsqueeze(-1)
u_train_tensor = torch.tensor(u_train_n, dtype=torch.float32).unsqueeze(-1)

a_test_tensor = torch.tensor(a_test_n, dtype=torch.float32).unsqueeze(-1)
u_test_tensor = torch.tensor(u_test_n, dtype=torch.float32).unsqueeze(-1)

dataset = TensorDataset(a_train_tensor, u_train_tensor)
loader = DataLoader(dataset, batch_size=20, shuffle=True)


# ============================================================
# 4. Train FNO
# ============================================================

model = FNO2d(modes1=12, modes2=12, width=32).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

epochs = 300

print("Training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for a_batch, u_batch in loader:
        a_batch = a_batch.to(device)
        u_batch = u_batch.to(device)

        pred = model(a_batch)
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

model.eval()
errors = []

with torch.no_grad():
    for i in range(num_test):
        a_input = a_test_tensor[i:i+1].to(device)

        pred = model(a_input).cpu().numpy()[0, :, :, 0]
        pred = pred * u_std + u_mean

        true = u_test[i]

        rel_error = np.linalg.norm(pred - true) / np.linalg.norm(true)
        errors.append(rel_error)

print(f"\nMean relative test error: {np.mean(errors):.4e}")
print(f"Median relative test error: {np.median(errors):.4e}")