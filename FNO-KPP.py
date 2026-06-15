import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer as timer

# device = 'cuda:6'
device = 'cpu' 
T = 1.0
steps = 100
dt = T / steps
sol = np.load('./data_gen/data/data_kpp.npy')
num_steps = 20
num_points = sol.shape[-1]
x = np.linspace(0, 1, num_points)
p = 64
M = 100
ntrain = 20
beta = np.zeros([sol.shape[0],sol.shape[1],p])
# beta1 = np.zeros([sol.shape[0],sol.shape[1],p])
for i in range(p):
    basis = sol*np.sin((i+1)*np.pi*x)
    beta[:,:,i] = np.mean(basis[...,:-1]+basis[...,1:],-1)
target = beta
myall = torch.tensor(target[:ntrain,:,:]).to(device)
initial = torch.tensor(target[:ntrain,0,:]).to(device)
target = torch.tensor(target[:ntrain,1:,:]).to(device)
eigen = (torch.linspace(1,p,p)*torch.pi).to(device)**2
ini = torch.tensor(sol[:ntrain,0,:]).to(device)
yall = torch.tensor(sol[:ntrain,:,:]).to(device)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cdouble))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cdouble)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width 
        # self.width = 48
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x   
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

# Initialize the neural network, optimizer, and loss function
epochs = 5000
lr = 1e-3
net = FNO1d(64,p).to(device)
print(count_params(net))
net = net.double()
optimizer = optim.Adam(net.parameters(), lr=lr)
step_size = 1000
gamma = 0.25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train_err = []
x = torch.tensor(x).to(device)
x1 = x.unsqueeze(-1)
f_co = (myall[:,1:num_steps+1,...]-myall[:,:num_steps,...])/dt + myall[:,1:num_steps+1,...]*eigen
fA = torch.zeros([ntrain,num_steps,num_points]).to(device)
for j in range(p):
    fA += f_co[...,j].unsqueeze(-1)*((torch.sin((j+1)*torch.pi*x)).unsqueeze(0).unsqueeze(0))
for i in range(epochs):
    t1 = timer()
    u0 = initial
    uini = ini
    loss = 0
    for step in range(num_steps):
        uui = net(uini.unsqueeze(-1))
        uu = torch.zeros([ntrain,p]).double().to(device)
        for j in range(p):
            basis = uui.squeeze(-1)*torch.sin((j+1)*torch.pi*x)
            uu[:,j] = torch.mean(basis[:,1:]+basis[:,:-1],-1)
        u = (u0 + dt*uu)/(1+dt*eigen)
        loss += torch.mean(torch.norm(u-target[:,step,:],2,-1)/torch.norm(target[:,step,:],2,-1))/num_steps
        Non = net(yall[:,step,:].unsqueeze(-1)).squeeze(-1)
        ffA = fA[:,step,...]
        loss += torch.mean(torch.norm(ffA - Non,2,-1)/torch.norm(ffA,2,-1))/num_steps
        u0 = u
        output = torch.zeros([ntrain,num_points]).double().to(device)
        for j in range(p):
                output += u0[...,j].unsqueeze(-1)*((torch.sin((j+1)*torch.pi*x)).unsqueeze(0))     
        uini = output
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_err.append(loss.item())
    t2 = timer()
    if (i+1) % 5 == 0:
        print(f"Step {i+1}, Loss: {loss.item()}",flush=True)
