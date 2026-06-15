import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer

# device = 'cuda:0'
device = 'cpu'
dt = 1/100
sol = np.load('./data_gen/data/data_kpp.npy')
num_steps = 20
num_points = sol.shape[-1]
x = np.linspace(0, 1, num_points)
p = 64
M = 1000
ntrain = 100 
beta = np.zeros([sol.shape[0],sol.shape[1],p])
for i in range(p):
    basis = sol*np.sin((i+1)*np.pi*x)
    beta[:,:,i] = np.mean(basis[...,:-1]+basis[...,1:],-1)
target = beta
myall = torch.tensor(target[:ntrain,:,:]).to(device)
initial = torch.tensor(target[:ntrain,0,:]).to(device)
target = torch.tensor(target[:ntrain,1:,:]).to(device)
eigen = (torch.linspace(1,p,p)*torch.pi).to(device)**2
class NonlinearNet(nn.Module):
    def __init__(self, M, p):
        super(NonlinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(p, M),
            nn.ReLU(),
            nn.Linear(M, M),
	    nn.ReLU(),
            nn.Linear(M, p)
        )

    def forward(self, u):
        return self.net(u)

epochs = 5000
lr = 1e-3
net = NonlinearNet(M,p).to(device)
net = net.double()
optimizer = optim.Adam(net.parameters(), lr=lr)
step_size = 1000
gamma = 0.25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train_err = []
t1 = 0
for i in range(epochs):
    u0 = initial
    loss = 0
    for step in range(num_steps):
        u = (u0 + dt*net(u0))/(1+dt*eigen)
        loss += torch.mean(torch.norm(u-target[:,step,:],2,-1)/torch.norm(target[:,step,:],2,-1))/num_steps
        Non = net(myall[:,step,:])
        ff = (myall[:,step+1,:]-myall[:,step,:])/dt + myall[:,step+1,:]*eigen
        loss += torch.mean(torch.norm(ff - Non,2,-1)/torch.norm(ff,2,-1))/num_steps
        u0 = u
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_err.append(loss.item())
    if (i+1) % 50 == 0:
        t2 = timer()
        print((t2-t1)/50)
        t1 = t2
        print(f"Step {i+1}, Loss: {loss.item()}")
with torch.no_grad():
    xx = torch.linspace(0,1,num_points).to(device)
    sol = torch.tensor(sol).to(device)
    u0 = initial
    loss = 0
    loss1 = 0
    net.eval()
    for step in range(num_steps):
        u = (u0 + dt*net(u0))/(1+dt*eigen)
        output = torch.zeros(ntrain,num_points).to(device)
        for j in range(p):
            output += u[:,j].unsqueeze(-1)*torch.sin((j+1)*torch.pi*xx).unsqueeze(0)
            Non = net(myall[:,step,:])
        ff = (myall[:,step+1,:]-myall[:,step,:])/dt + myall[:,step+1,:]*eigen
        loss1 += torch.mean(torch.norm(ff - Non,2,-1)/torch.norm(ff,2,-1))/num_steps
        u0 = u
    tt = torch.tensor((1.-sol)*sol).to(device)
    bb = torch.tensor(beta).to(device)
    u = net(bb)
    output = torch.zeros_like(tt)
    for j in range(p):
        output += u[:,:,j].unsqueeze(-1)*torch.sin((j+1)*torch.pi*xx).unsqueeze(0)
    loss2 = torch.mean(torch.norm(output[:ntrain,:num_steps,:]-tt[:ntrain,:num_steps,:],2,-1)/torch.norm(tt[:ntrain,:num_steps,:],2,-1))
    print("L2 loss: {:2.2e}, Residual loss: {:2.2e}, Nonlinear loss: {:2.2e}".format(loss,loss1,loss2))
