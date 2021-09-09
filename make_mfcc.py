#!/usr/bin/env python

import os

import numpy as np
import torch
import torchaudio
from speechbrain.lobes import features
from tqdm import tqdm
import argparse


def make_mfcc_from_paths(path_file, out_path):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    fmaker = features.MFCC(deltas=False, context=False, n_mels=35, n_mfcc=30, n_fft=512)
    i = 0
    with open(path_file) as f:
        wav_paths, speakers = zip(*[line.split() for line in f.readlines()])
    unique_speakers = np.unique(speakers).tolist()
    speakers = np.array(list(map(lambda s: unique_speakers.index(s), speakers)))
    for wav_path in tqdm(wav_paths):
        sig, rate = torchaudio.load(wav_path)
        mfccname = f'mfcc{i:07d}.pt'
        torch.save(fmaker(sig), os.path.join(out_path, mfccname))
        i += 1
    np.save(os.path.join(out_path, 'speakers.npy'), speakers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to list of wav paths and speaker labels",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to MFCC folder",
        required=True,
    )
    args = parser.parse_args()

    make_mfcc_from_paths(args.input, args.output)

