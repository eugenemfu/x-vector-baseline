import numpy as np
import torch
from skimage import transform


def specaug(spec: torch.tensor):
    new_size = max(spec.shape[1] - spec.shape[1] // 4, int(np.random.normal(spec.shape[1], 0.05 * spec.shape[1])))
    spec = torch.tensor(transform.resize(spec.squeeze(0), (new_size, spec.shape[2]))).unsqueeze(0)
    for _ in range(2):
        dt = np.random.randint(0, min(100, spec.shape[1] // 2))
        t0 = np.random.randint(0, spec.shape[1] - dt)
        spec[:, t0:t0+dt, :] = torch.zeros((spec.shape[0], dt, spec.shape[2]))
        df = np.random.randint(0, 8)
        f0 = np.random.randint(0, spec.shape[2] - df)
        spec[:, :, f0:f0+df] = torch.zeros((spec.shape[0], spec.shape[1], df))
    return spec


class DataLoader:
    def __init__(self,
                 data_path,
                 y,
                 idx, 
                 batch_size=64, 
                 maxlen=2500,
                 device='cuda',
                 augment=False,
                ):
        self.idx = idx
        self.batch_size = batch_size
        self.data_path = data_path
        self.y = y
        self.maxlen = maxlen
        self.device = device
        self.augment = augment

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
                try:
                    sample = torch.load(f'{self.data_path}/mfcc{i:07d}.pt')
                except FileNotFoundError:
                    sample = torch.load(f'{self.data_path}/mfcc{i}')
                if self.augment:
                    sample = specaug(sample)
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