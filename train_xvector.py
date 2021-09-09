import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import argparse
from datetime import datetime
import logging
logging.basicConfig(filename='log/train_xvector.log', level=logging.INFO)

from utils.xvector import ETDNN_xvector
from utils.dataloader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    type=str,
    help="Path to MFCC folder",
    required=True,
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    help="Batch size",
    required=False,
    default=16,
)
parser.add_argument(
    "-e",
    "--epoch_num",
    type=int,
    help="Number of epochs",
    required=False,
    default=100,
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="Learning rate",
    required=False,
    default=1e-5,
)
parser.add_argument(
    "-m",
    "--load_model",
    type=int,
    help="Number of epoch to start from if the model was saved",
    required=False,
    default=0,
)
args = parser.parse_args()


device = torch.device("cuda:1")

SEED = 0
np.random.seed(SEED)


def eval_accuracy(model, data_val, num_batches=None):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    losses = []
    top1_accs = []
    # top3_accs = []
    for inputs, targets, lengths in data_val():
        with torch.no_grad():
            output = model.classify(inputs, lengths)
            loss = criterion(output, targets)
            losses.append(loss.item())
            y_top1 = np.argmax(output.cpu().detach().numpy(), axis=1)
            # y_top3 = np.argpartition(output.cpu().detach().numpy(), -3, axis=1)[:, -3:]
            top1_accs.append(np.mean(targets.cpu().detach().numpy() == y_top1))
        if num_batches:
            num_batches -= 1
            if num_batches == 0:
                break
    loss = np.mean(losses)
    top1_acc = np.mean(top1_accs)
    model.train()
    return loss, top1_acc

def model_filename(number):
    return f'models/model_mix_{number}.pkl'

def logstring(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc):
    return (str(datetime.now())[:-7] + " --- " +
        f'epoch: {epoch:02d}/{num_epochs}, ' +
        f'train loss: {"%.5f"%train_loss}, train acc: {"%.5f"%train_acc}, ' +
        f'val loss: {"%.5f"%val_loss}, val acc: {"%.5f"%val_acc}')


data_path = args.input
y = np.load(data_path + '/speakers.npy')
assert len(np.unique(y)) == np.max(y) + 1

data_size = y.shape[0]

train_idx = np.array([], dtype=int)
val_idx = np.array([], dtype=int)

try:
    train_idx = np.load('lists/mix_train_idx.npy')
    val_idx = np.load('lists/mix_val_idx.npy')

except FileNotFoundError:
    val_size = 0.2
    last_class = 0
    last_class_start_point = 0
    for i in range(data_size + 1):
        if i == data_size or y[i] != last_class:
            class_size = i - last_class_start_point
            rand_idx = last_class_start_point + np.random.choice(class_size, size=class_size, replace=False)
            border = int(val_size * class_size)
            train_idx = np.concatenate((train_idx, rand_idx[border:]))
            val_idx = np.concatenate((val_idx, rand_idx[:border]))
            if i != data_size:
                last_class = y[i]
                last_class_start_point = i
    np.save(f'lists/mix_train_idx.npy', train_idx)
    np.save(f'lists/mix_val_idx.npy', val_idx)


batch_size = args.batch_size
num_epochs = args.epoch_num

data_train = DataLoader(data_path, y, train_idx,
                        batch_size, device=device, augment=True)
data_train_eval = DataLoader(data_path, y, train_idx, batch_size, device=device)
data_val = DataLoader(data_path, y, val_idx, batch_size, device=device)

n_speakers = len(np.unique(y))
first_epoch = args.load_model
if first_epoch == 0:
    model = ETDNN_xvector(n_speakers).to(device)
else:
    model = torch.load(model_filename(first_epoch)).to(device)

optimizer = Adam(model.parameters(), lr=args.learning_rate) #, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss().to(device)

logging.info(str(datetime.now())[:-7] + " --- training started.")
print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

print(f'\repoch: {first_epoch}/{num_epochs}, evaluating...', end='')
train_loss, train_acc = eval_accuracy(model, data_train_eval, num_batches=200)
val_loss, val_acc = eval_accuracy(model, data_val, num_batches=200)

log = logstring(first_epoch, num_epochs, train_loss, train_acc, val_loss, val_acc)
print("\r" + log)
logging.info(log)

for epoch in range(first_epoch, num_epochs):
    losses = []
    i = 0
    best_loss = 5
    for inputs, targets, lengths in data_train():
        i += 1
        outputs = model.classify(inputs, lengths)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f'\repoch: {epoch+1}/{num_epochs}, batch: {i}/{train_idx.shape[0]//batch_size+1}, size: {inputs.shape[1]}', end='')
    if (epoch + 1) % 1 == 0:
        print(f'\repoch: {epoch+1}/{num_epochs}, train loss: {"%.5f"%np.mean(losses)}, evaluating...', end='')
        train_loss, train_acc = eval_accuracy(model, data_train_eval, num_batches=2000)
        val_loss, val_acc = eval_accuracy(model, data_val, num_batches=2000)
        saved = False
        if val_loss < best_loss:
            saved = True
            best_loss = val_loss
            torch.save(model, model_filename(epoch+1))

        log = logstring(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc)
        if saved:
               log += f', saved as {model_filename(epoch+1)}'

        print('\r' + log)
        logging.info(log)
