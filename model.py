import torch
import torch.nn as nn
import torch.nn.functional as F
from tdnn import TDNN

class Net(nn.Module):
    def __init__(self, n_speakers):
        super().__init__()
        hidden_size = 512
        xvector_size = 512
        pool_size = 1500
        dropout = 0.2
        self.tdnn1 = TDNN(30, hidden_size, 5, dropout_p=dropout)
        self.tdnn2 = TDNN(hidden_size, hidden_size, 3, dilation=2, dropout_p=dropout)
        self.tdnn3 = TDNN(hidden_size, hidden_size, 3, dilation=3, dropout_p=dropout)
        self.fc5 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.fc6 = nn.Sequential(nn.Linear(hidden_size, pool_size), nn.ReLU())
        self.emb = nn.Linear(2 * pool_size, xvector_size)
        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.Linear(xvector_size, xvector_size),
            nn.ReLU(),
            nn.Linear(xvector_size, n_speakers))

    def embedding(self, x, lengths=None):
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.fc5(x)
        x = self.fc6(x)
        if lengths:
            assert x.shape[0] == len(lengths)
            maxlen = x.shape[1]
            stat = []
            for i in range(x.shape[0]):
                means = torch.mean(x[i, 0:int(maxlen*lengths[i]), :], 0)
                stds = torch.std(x[i, 0:int(maxlen*lengths[i]), :], 0)
                meanstd = torch.cat([means, stds], 0).unsqueeze(0)
                #print(meanstd.shape)
                stat.append(meanstd)
            stat = torch.cat(stat, 0)
        else:
            stat = torch.cat((torch.mean(x, 1), torch.std(x, 1)), 1)
        emb = self.emb(stat)
        return emb

    def forward(self, x, lengths=None):
        emb = self.embedding(x, lengths)
        res = self.tail(emb)
        return res
    

