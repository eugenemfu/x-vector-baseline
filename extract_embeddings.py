import numpy as np
import torch
from tqdm import tqdm
import argparse


def extract_embeddings(model, mfcc_folder, output_file, val_idx_file=''):
    speakers = np.load(mfcc_folder + '/speakers.npy')
    n = speakers.shape[0]
    if val_idx_file:
        val_idx = np.load(val_idx_file)
    else:
        val_idx = np.arange(0, n)
    embeddings = np.zeros((n, 512))
    for i in tqdm(val_idx):
        try:
            mfcc = torch.load(f'{mfcc_folder}/mfcc{i:07d}.pt').to(device)
        except FileNotFoundError:
            mfcc = torch.load(f'{mfcc_folder}/mfcc{i}').to(device)
        emb = model(mfcc).cpu().detach().numpy().squeeze(0)
        embeddings[i] = emb
    np.savez(output_file, embeddings=embeddings[val_idx], speakers=speakers[val_idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to embedding pkl model",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to MFCC folder",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to embeddings+labels .npz file",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--validx",
        type=str,
        help="Path to .npy file with indices of elements to consider",
        required=False,
        default=''
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")

    SEED = 1000
    np.random.seed(SEED)

    model = torch.load(args.model).to(device)
    model.eval()

    extract_embeddings(model, args.input, args.output, args.validx)
# extract_embeddings('../mfcc/voxceleb2_dev', 'lists/embeddings_voxceleb2_val_53.npy')
# extract_embeddings('../mfcc/voxceleb1_test', 'lists/embeddings_voxceleb1_test_53.npy')