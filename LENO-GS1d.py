import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# device = 'cuda:4'
device = 'cpu' 
dt = .1
num_steps = 20
data = np.load('data_gen/data/data_GS.npz')
solA = data['resA']
solS = data['resS']
DA = 2.5e-4
DS = 5e-4
num_points = solA.shape[-1]
x = np.linspace(0, 2*np.pi, num_points)
p = 64
M = 1000
ntrain = solA.shape[0]
betaA = np.zeros([solA.shape[0],solA.shape[1],p])
betaS = np.zeros([solS.shape[0],solS.shape[1],p])
for i in range(p):
    if i == 0:
        betaA[:,:,i] = np.einsum('ijk,k',solA[:,:,:-1],np.cos((i)*x[:-1]/2))/(num_points-1)/2 + np.einsum('ijk,k',solA[:,:,1:],np.cos((i)*x[1:]/2))/(num_points-1)/2
        betaS[:,:,i] = np.einsum('ijk,k',solS[:,:,:-1],np.cos((i)*x[:-1]/2))/(num_points-1)/2 + np.einsum('ijk,k',solS[:,:,1:],np.cos((i)*x[1:]/2))/(num_points-1)/2
    else:
        betaA[:,:,i] = np.einsum('ijk,k',solA[:,:,:-1],np.cos((i)*x[:-1]/2))/(num_points-1) + np.einsum('ijk,k',solA[:,:,1:],np.cos((i)*x[1:]/2))/(num_points-1)
        betaS[:,:,i] = np.einsum('ijk,k',solS[:,:,:-1],np.cos((i)*x[:-1]/2))/(num_points-1) + np.einsum('ijk,k',solS[:,:,1:],np.cos((i)*x[1:]/2))/(num_points-1)
targetA = betaA
targetS = betaS
myallA = torch.tensor(targetA[:ntrain,:,:]).to(device)
initialA = torch.tensor(targetA[:ntrain,0,:]).to(device)
targetA = torch.tensor(targetA[:ntrain,1:,:]).to(device)
myallS = torch.tensor(targetS[:ntrain,:,:]).to(device)
initialS = torch.tensor(targetS[:ntrain,0,:]).to(device)
targetS = torch.tensor(targetS[:ntrain,1:,:]).to(device)
eigen = (torch.linspace(0,p-1,p)).to(device)**2/4
class NonlinearNet(nn.Module):
    def __init__(self, M, p):
        super(NonlinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2*p, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, 2*p)
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
for i in range(epochs):
    u0A = initialA
    u0S = initialS
    u0 = torch.cat((u0A,u0S),dim=-1)
    loss = 0
    for step in range(num_steps):
        fcoe = net(u0)
        uA = (u0A + dt*fcoe[:,0:p])/(1+DA*dt*eigen)
        uS = (u0S + dt*fcoe[:,p:])/(1+DS*dt*eigen)
        loss += torch.mean(torch.norm(uA-targetA[:,step,:],2,-1)/(torch.norm(targetA[:,step,:],2,-1)))/num_steps
        loss += torch.mean(torch.norm(uS-targetS[:,step,:],2,-1)/(torch.norm(targetS[:,step,:],2,-1)))/num_steps
        ffA = (myallA[:,step+1,:]-myallA[:,step,:])/dt + DA*myallA[:,step+1,:]*eigen
        ffS = (myallS[:,step+1,:]-myallS[:,step,:])/dt + DS*myallS[:,step+1,:]*eigen
        myall = torch.cat((myallA[:,step,:],myallS[:,step,:]),dim=-1)
        Non = net(myall)
        loss += torch.mean(torch.norm(ffA - Non[:,0:p],2,-1)/(torch.norm(ffA,2,-1)))/num_steps
        loss += torch.mean(torch.norm(ffS - Non[:,p:],2,-1)/(torch.norm(ffS,2,-1)))/num_steps
        u0A = uA
        u0S = uS
        u0 = torch.cat((u0A,u0S),dim=-1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_err.append(loss.item())
    if (i+1) % 50 == 0:
        print(f"Step {i+1}, Loss: {loss.item()}")
with torch.no_grad():
    solA = torch.tensor(solA).to(device)
    solS = torch.tensor(solS).to(device)
    net.eval()
    u0A = initialA
    u0S = initialS
    u0 = torch.cat((u0A,u0S),dim=-1)
    lossA = 0
    lossS = 0
    loss1A = 0
    loss1S = 0
    xx = torch.linspace(0,2*np.pi,num_points).to(device)
    for step in range(num_steps):
        fcoe = net(u0)
        uA = (u0A + dt*fcoe[:,0:p])/(1+DA*dt*eigen)
        uS = (u0S + dt*fcoe[:,p:])/(1+DS*dt*eigen)
        outputA1 = torch.zeros(ntrain,num_points).to(device)
        outputS1 = torch.zeros(ntrain,num_points).to(device)
        for j in range(p):
            outputA1 += uA[:,j].unsqueeze(-1)*torch.cos((j)*xx/2).unsqueeze(0)
            outputS1 += uS[:,j].unsqueeze(-1)*torch.cos((j)*xx/2).unsqueeze(0)
        lossA += torch.mean(torch.norm((outputA1-solA[:,step+1,:]),2,-1)/torch.norm(solA[:,step+1,:],2,-1))/num_steps
        lossS += torch.mean(torch.norm((outputS1-solS[:,step+1,:]),2,-1)/torch.norm(solS[:,step+1,:],2,-1))/num_steps
        ffA = (myallA[:,step+1,:]-myallA[:,step,:])/dt + DA*myallA[:,step+1,:]*eigen
        ffS = (myallS[:,step+1,:]-myallS[:,step,:])/dt + DS*myallS[:,step+1,:]*eigen
        myall = torch.cat((myallA[:,step,:],myallS[:,step,:]),dim=-1)
        Non = net(myall)
        loss1A += torch.mean(torch.norm(ffA - Non[:,0:p],2,-1)/(torch.norm(ffA,2,-1)))/num_steps
        loss1S += torch.mean(torch.norm(ffS - Non[:,p:],2,-1)/(torch.norm(ffS,2,-1)))/num_steps
        u0A = uA
        u0S = uS
        u0 = torch.cat((u0A,u0S),dim=-1)
    A = solA[:ntrain,:num_steps+1,:]
    S = solS[:ntrain,:num_steps+1,:]
    mu = .065
    rho = .04
    fA = torch.tensor(S*A**2-(mu+rho)*A).to(device)
    fS = torch.tensor(-S*A**2+rho*(1.-S)).to(device)
    betaA = torch.tensor(betaA).to(device)
    betaS = torch.tensor(betaS).to(device)
    beta = torch.cat((betaA[:ntrain,:num_steps+1,:],betaS[:ntrain,:num_steps+1,:]),dim=-1)
    u = net(beta)
    xx = torch.linspace(0,2*torch.pi,num_points).to(device).double()
    outputA = torch.zeros_like(fA)
    outputS = torch.zeros_like(fS)
    for j in range(p):
        outputA += u[:,:,j].unsqueeze(-1)*torch.cos((j)*xx/2).unsqueeze(0)
        outputS += u[:,:,j+p].unsqueeze(-1)*torch.cos((j)*xx/2).unsqueeze(0)
    loss2A = torch.mean(torch.norm(outputA-fA,2,-1))/num_points*torch.pi*2
    loss3A = torch.mean(torch.norm(outputA-fA,2,-1)/torch.norm(fA,2,-1))
    loss2S = torch.mean(torch.norm(outputS-fS,2,-1))/num_points*torch.pi*2
    loss3S = torch.mean(torch.norm(outputS-fS,2,-1)/torch.norm(fS,2,-1))
    print("A: L2 loss: {:2.2e}, Residual loss: {:2.2e}, Nonlinear loss Absolute: {:2.2e},  Nonlinear loss Relative: {:2.2e}".format(lossA,loss1A,loss2A,loss3A))
    print("S: L2 loss: {:2.2e}, Residual loss: {:2.2e}, Nonlinear loss Absolute: {:2.2e},  Nonlinear loss Relative: {:2.2e}".format(lossS,loss1S,loss2S,loss3S))
