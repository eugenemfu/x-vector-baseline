#!/usr/bin/env python

import numpy as np
import pickle

SEED = 1000
np.random.seed(SEED)

archive = np.load('lists/emb_mix_val_mix59.npz')
embeddings, speakers = archive.values()
train_size = speakers.shape[0]
print(train_size)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=200)
embeddings_reduced = lda.fit_transform(embeddings, speakers)

from plda.model import Model as PLDA
plda = PLDA(embeddings_reduced, speakers)

with open('models/plda_mix59.pkl', 'wb') as f:
    pickle.dump((lda, plda), f)






