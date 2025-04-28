#!/usr/bin/env python3
import os
import sys
import signal

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import wandb
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch.cuda.amp import autocast
from torch.amp import GradScaler


# -----------------------
# 1) CONFIGURATION
# -----------------------
EPOCHS      = 500
BATCH_SIZE  = 64
LR          = 1e-2
NUM_CLASSES = 40

TRAIN_H5  = 'train.h5'
TEST_H5   = 'test.h5'
SAVE_DIR  = 'checkpoints'
PLOTS_DIR = 'plots'
LATEST_CKPT = os.path.join(SAVE_DIR, 'checkpoint_epoch_12.pth')

os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------
# 2) DATASET
# -----------------------
class H5Lazy(torch.utils.data.Dataset):
    """Lazy HDF5: one file-handle per worker."""
    def __init__(self, path):
        self.path = path
        with h5py.File(self.path, 'r') as hf:
            self.n = hf['data'].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if not hasattr(self, 'hf'):
            self.hf = h5py.File(self.path, 'r')
        pts   = self.hf['data'][idx]
        label = int(self.hf['label'][idx])
        return Data(
            pos=torch.from_numpy(pts).float(),
            y=torch.tensor(label, dtype=torch.long),
        )


# -----------------------
# 3) MODEL
# -----------------------
class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, output_channels=NUM_CLASSES):
        super().__init__()
        self.conv1 = DynamicEdgeConv(
            nn.Sequential(nn.Linear(6,64), nn.ReLU(), nn.Linear(64,64)), k=k)
        self.conv2 = DynamicEdgeConv(
            nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,64)), k=k)
        self.conv3 = DynamicEdgeConv(
            nn.Sequential(nn.Linear(128,128), nn.ReLU(), nn.Linear(128,128)), k=k)
        self.conv4 = DynamicEdgeConv(
            nn.Sequential(nn.Linear(256,256), nn.ReLU(), nn.Linear(256,256)), k=k)

        self.lin1 = nn.Linear(512, emb_dims)
        self.bn1  = nn.BatchNorm1d(emb_dims)
        self.dp1  = nn.Dropout(dropout)
        self.lin2 = nn.Linear(emb_dims, output_channels)

    def forward(self, data):
        x, batch = data.pos, data.batch
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        xpool = global_max_pool(x_cat, batch)

        x = F.relu(self.bn1(self.lin1(xpool)))
        x = self.dp1(x)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


# -----------------------
# 4) UTILITIES
# -----------------------
def save_confusion_matrix(cm, epoch):
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — Epoch {epoch}')
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]),
                    ha='center',
                    color='white' if cm[i,j]>thresh else 'black')
    path = os.path.join(PLOTS_DIR, f'cm_epoch_{epoch:03d}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# -----------------------
# 5) TRAIN & TEST STEPS
# -----------------------
def train_one_epoch(model, opt, loader, device, scaler, epoch, global_step):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=f"[{epoch}/{EPOCHS}] Train", unit="batch"):
        batch = batch.to(device, non_blocking=True)
        opt.zero_grad()
        with autocast():
            out  = model(batch)
            loss = F.nll_loss(out, batch.y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        ls = loss.item()
        total_loss += ls * batch.num_graphs
        global_step += 1
        wandb.log({'train/loss': ls}, step=global_step)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, global_step


def test_one_epoch(model, loader, device, epoch):
    model.eval()
    total_loss = 0.0
    correct    = 0
    all_y, all_p, all_prob = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{epoch}/{EPOCHS}]  Test", unit="batch"):
            batch = batch.to(device, non_blocking=True)
            out   = model(batch)
            loss  = F.nll_loss(out, batch.y, reduction='sum')
            probs = out.exp()
            preds = probs.argmax(dim=1)

            total_loss += loss.item()
            correct    += int((preds==batch.y).sum())
            all_y.append(batch.y.cpu().numpy())
            all_p.append(preds.cpu().numpy())
            all_prob.append(probs.detach().cpu().numpy())

            wandb.log({'test/loss': loss.item()/batch.num_graphs})

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    y_prob = np.concatenate(all_prob)

    loss_epoch = total_loss / len(loader.dataset)
    acc_epoch  = correct    / len(loader.dataset)
    f1_epoch   = f1_score(y_true, y_pred, average='macro')
    y_bin      = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
    roc_epoch  = roc_auc_score(y_bin, y_prob,
                              multi_class='ovr', average='macro')
    cm = confusion_matrix(y_true, y_pred)
    cm_path = save_confusion_matrix(cm, epoch)

    wandb.log({
        'test/epoch_loss':       loss_epoch,
        'test/accuracy':         acc_epoch,
        'test/f1':               f1_epoch,
        'test/roc_auc':          roc_epoch,
        'test/confusion_matrix': wandb.Image(cm_path)
    }, step=epoch)

    return loss_epoch, acc_epoch


# -----------------------
# 6) MAIN
# -----------------------
if __name__ == "__main__":
    # W&B init
    wandb.init(
        project="modelnet40-dgcnn",
        name=f"run_bs{BATCH_SIZE}_lr{LR}",
        config={'epochs':EPOCHS,'bs':BATCH_SIZE,'lr':LR}
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    torch.backends.cudnn.benchmark = True

    # data loaders with 16 workers
    train_loader = DataLoader(
        H5Lazy(TRAIN_H5),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        H5Lazy(TEST_H5),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # model/opt/scaler
    model     = DGCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    # resume if a checkpoint exists
    start_epoch = 1
    global_step = 0
    if os.path.exists(LATEST_CKPT):
        ckpt = torch.load(LATEST_CKPT, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', 0)
        print(f"Resuming from epoch {start_epoch}")

    # graceful Ctrl+C save
    def _save_and_exit(sig, frame):
        print("\nInterrupted → saving latest.pth")
        torch.save({
            'epoch': start_epoch,
            'global_step': global_step,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, LATEST_CKPT)
        sys.exit(0)
    signal.signal(signal.SIGINT, _save_and_exit)

    # training loop
    for epoch in range(start_epoch, EPOCHS+1):
        tr_loss, global_step = train_one_epoch(
            model, optimizer, train_loader,
            device, scaler, epoch, global_step
        )
        print(f"Epoch {epoch:03d} | Train Loss: {tr_loss:.4f}")

        te_loss, te_acc = test_one_epoch(
            model, test_loader, device, epoch
        )
        print(f"           Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.4f}")

        # checkpoint per‑epoch and latest
        ckpt = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(ckpt,
                   os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch}.pth'))
        torch.save(ckpt, LATEST_CKPT)

    # final save
    torch.save(model.state_dict(), 'model_final.pth')
    print("Training complete. Saved model_final.pth")
    wandb.finish()
