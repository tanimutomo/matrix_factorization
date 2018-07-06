import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import _pickle
import datetime
import pandas as pd

# _pickle import
with open('pkls/zansa_12.pkl', 'rb') as f:
    df = _pickle.load(f)
with open('pkls/mesh12_list.pkl', 'rb') as f:
    mesh12_list = _pickle.load(f)
with open('pkls/fm_list.pkl', 'rb') as f:
    fm_list = _pickle.load(f)
with open('pkls/tsdiv_list.pkl', 'rb') as f:
    tsdiv_list = _pickle.load(f)
with open('pkls/kind_list.pkl', 'rb') as f:
    kind_list = _pickle.load(f)
with open('pkls/condition_list.pkl', 'rb') as f:
    condition_list = _pickle.load(f)
with open('pkls/not_condition_list.pkl', 'rb') as f:
    ncondition_list = _pickle.load(f)
with open('pkls/other_index.pkl', 'rb') as f:
    others_index = _pickle.load(f)
with open('pkls/mesh_list.pkl', 'rb') as f:
    mesh_list = _pickle.load(f)

class TF(nn.Module):
    def __init__(self, input_size, R):
        super(TF, self).__init__()
        self.I = input_size[0]
        self.J = input_size[1]
        self.K = input_size[2]
        self.U = nn.Parameter(torch.Tensor(R, self.I).uniform_(0,1), requires_grad=True)
        self.V = nn.Parameter(torch.Tensor(R, self.J).uniform_(0,1), requires_grad=True)
        self.W = nn.Parameter(torch.Tensor(R, self.K).uniform_(0,1), requires_grad=True)
        
    def non_negative(self, R=None):
#         torch.clamp(self.U, min=0)
#         torch.clamp(self.V, min=0)
#         torch.clamp(self.W, min=0)
        torch.clamp(self.U, min=0)
        torch.clamp(self.V, min=0)
        torch.clamp(self.W, min=0)

    def forward_one_rank(self, u, v, w):
        UV = torch.ger(u, v)
        UV2 = UV.unsqueeze(2).repeat(1,1,self.K)
        W2 = w.unsqueeze(0).unsqueeze(1).repeat(self.I, self.J, 1)
        outputs = UV2 * W2
        return outputs
    
    def forward(self, X=None):
        if X is not None:
            weight = (X != 0).float()
        else:
            weight = Variable(torch.ones(self.I, self.J, self.K))
        
        output = self.forward_one_rank(self.U[0], self.V[0], self.W[0])
        
        for i in np.arange(1, R):
            one_rank = self.forward_one_rank(self.U[i], self.V[i], self.W[i])
            output = output + one_rank
        
        return output

def mse_loss(output, df, tslist, meshlist, condlist, ncondlist, othlist):
    df1 = df.loc[othlist]
    bar = '-'
    loss = 0
    loss_all = 0
    for i in range(len(condlist)):
        print(bar)
        bar += '-'
        if ncondlist[i] == '':
            df2 = df.loc[list(df[df.text.str.contains(condlist[i])].index)]
        elif ncondlist[i] != '':
            a = list(df[df.text.str.contains(condlist[i])].index)
            b = list(df[df.text.str.contains(ncondlist[i])].index)
            c = list(set(a) - set(b))
            df2 = df.loc[c]
            
        for j in range(len(tslist)):
            ts_condition_from = df2.ts >= tslist[j]
            if j != 61:
                ts_condition_to = df2.ts < tslist[j+1]
                
            for k in range(len(meshlist)):
                mesh_condition = df2.mesh12 == meshlist[k]
                if k == len(tslist):
                    val = len(df2[(ts_condition_from) & (mesh_condition)].index)
                else:
                    val = len(df2[(ts_condition_from) & (ts_condition_to) & (mesh_condition)].index)
                loss = output[i, j, k] - val
                loss_all += loss
                loss = 0
                val = 0
                
            if i == len(tslist):
                val = len(df2[(ts_condition_from) & (mesh_condition)].index)
            else:
                val = len(df2[(ts_condition_from) & (ts_condition_to) & (mesh_condition)].index)
            loss = output[27,j,k] - val
            loss_all += loss
            loss = 0
            val = 0
        print(loss_all)

input_size = (28, 62, 355)
# X = torch.Tensor(62, 355, 28).uniform_(0, 1)

R = 2
model = TF(input_size, R)
# criterion = torch.nn.MSELoss(size_average=False)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# X = Variable(X)

epoch = 1000
for i in range(epoch):
    output = model.forward()
    loss = mse_loss(output, df, tsdiv_list, mesh_list, condition_list, ncondition_list, others_index)
    
    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()
    
    model.non_negative()
    
    if i % 100 == 0:
        print('{}: {}'.format(i, loss.data[0]))
        
print('{}: {}'.format(epoch, loss.data[0]))
print(model.forward().shape)


