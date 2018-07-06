import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import math

class MF(nn.Module):
    def __init__(self, input_size, K): 
        super(MF,self).__init__()
        self.input_size = input_size
        self.P = nn.Parameter(torch.Tensor(K, input_size[0]).uniform_(0,1))
        self.Q = nn.Parameter(torch.Tensor(K, input_size[1]).uniform_(0,1))

    def forward(self):
        return torch.matmul(torch.t(self.P), self.Q) 

    def project(self):
        self.P.data = torch.max(self.P.data, torch.zeros(self.P.data.size()))
        self.Q.data = torch.max(self.Q.data, torch.zeros(self.Q.data.size()))


def ll_gaussian(x, mu, sigma2, missing=False):
    if missing:
        weight = (x != 0).float()
        return -0.5 * (math.log(2 * math.pi * sigma2) + torch.pow(x - mu, 2) / sigma2) * weight
    return -0.5 * (math.log(2 * math.pi * sigma2) + torch.pow(x - mu, 2) / sigma2)


def ll_poisson(x, lam, missing=False):
    if missing:
        weight = (x != 0).float()
        return (x * torch.log(lam) - lam) * weight
    return x * torch.log(lam) - lam


def update_param_gaussian(R, model, optimizer, missing=False):
    optimizer.zero_grad()
    pred = model()
    log_likelihood = ll_gaussian(R, pred, 1, missing=missing)
    prior_P = ll_gaussian(model.P, 0, 1/0.003)
    prior_Q = ll_gaussian(model.Q, 0, 1/0.003)
    loss = -(torch.sum(log_likelihood) + torch.sum(prior_P) + torch.sum(prior_Q))
    loss.backward()
    optimizer.step()
    return loss.data[0]

def update_param_poisson(R, model, optimizer, missing=False):
    optimizer.zero_grad()
    pred = model()
    log_likelihood = ll_poisson(R, pred, missing=missing)
    loss = -torch.sum(log_likelihood)
    loss.backward()
    optimizer.step()
    return loss.data[0]

if __name__ == '__main__':
    R = Variable(torch.FloatTensor([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4],
            ]
        ))

    model = MF(R.size(), 2)
    optimizer_p = optim.Adam([model.P], lr = 0.02)
    optimizer_q = optim.Adam([model.Q], lr = 0.02)

    for i in range(1000):
        update_param_poisson(R, model, optimizer_p, missing=True)
        model.project()
        loss = update_param_poisson(R, model, optimizer_q, missing=True)
        model.project()

        if i % 100 == 0:
            print("Epoch {}: Loss:{}".format(i+1, loss))

    print("Epoch {}: Loss:{}".format(i+1, loss))
    print("Ground Truth:{}".format(R.data))
    print("Predict:{}".format(model().data))

    print("P:{}".format(model.P.data))
    print("Q:{}".format(model.Q.data))

