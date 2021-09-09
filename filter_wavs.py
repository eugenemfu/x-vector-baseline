import sys

import torch
import torchaudio
from tqdm import tqdm

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

wav_paths, speakers = zip(*[line.split() for line in lines])

out_lines = []

for i in tqdm(range(len(lines))):
    sig, rate = torchaudio.load(wav_paths[i])
    if sig.shape[1] / rate < 2:
        continue
    out_lines.append(lines[i])

with open(sys.argv[2], 'w') as f:
    f.writelines(out_lines)
