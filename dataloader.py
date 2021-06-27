import numpy as np
import torch


class DataLoader:
    def __init__(self,
                 data_path, 
                 file_name,
                 y,
                 idx, 
                 batch_size=64, 
                 maxlen=2500,
                 device='cuda',
                ):
        self.idx = idx
        self.batch_size = batch_size
        self.data_path = data_path
        self.file_name = file_name
        self.y = y
        self.maxlen = maxlen
        self.device = device

    def next(self):
        m = self.idx.shape[0]
        rand_idxidx = np.random.choice(m, size=m, replace=False)
        idx = self.idx[rand_idxidx]
        pos = 0
        while pos < m:
            idx_batch = idx[pos:pos+self.batch_size]
            X_batch = []
            lengths = []
            max_len = 0
            for i in idx_batch:
                sample = torch.load(self.data_path + '/' + self.file_name + str(i))
                #print(sample.shape)
                if sample.shape[1] > self.maxlen:
                    sample = sample[:, :self.maxlen, :]
                #print(sample.shape)
                length = sample.shape[1]
                if length > max_len:
                    max_len = length
                X_batch.append(sample)
            for i in range(len(X_batch)):
                length = X_batch[i].shape[1]
                lengths.append(length / max_len)
                X_batch[i] = torch.cat([X_batch[i], torch.zeros(1, max_len - length, 30)], dim=1)
            X_batch = torch.cat(X_batch, dim=0)
            y_batch = self.y[idx_batch]
            yield X_batch.to(self.device), torch.tensor(y_batch, device=self.device), lengths
            pos += self.batch_size

    def __call__(self):
        return self.next()