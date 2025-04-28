#!/usr/bin/env python3
import os
import re
import h5py
import torch
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Force non‑interactive Agg backend so matplotlib never tries to spawn Tk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#  CONFIGURATION
# -----------------------------------------------------------------------------
CHECKPOINT_DIR = r"checkpoints_augmented_2x"
VAL_H5         = r"aug_data_split_2x\val.h5"
BATCH_SIZE     = 64
NUM_CLASSES    = 40
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR        = "val_results"
EPOCHS_DIR     = os.path.join(OUT_DIR, "epochs")
GLOBAL_DIR     = os.path.join(OUT_DIR, "global")
os.makedirs(EPOCHS_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
#  MODEL DEFINITION
# -----------------------------------------------------------------------------
class DGCNN(torch.nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, output_channels=NUM_CLASSES):
        super().__init__()
        self.conv1 = DynamicEdgeConv(
            torch.nn.Sequential(torch.nn.Linear(6,64),
                                torch.nn.ReLU(),
                                torch.nn.Linear(64,64)),
            k=k
        )
        self.conv2 = DynamicEdgeConv(
            torch.nn.Sequential(torch.nn.Linear(128,64),
                                torch.nn.ReLU(),
                                torch.nn.Linear(64,64)),
            k=k
        )
        self.conv3 = DynamicEdgeConv(
            torch.nn.Sequential(torch.nn.Linear(128,128),
                                torch.nn.ReLU(),
                                torch.nn.Linear(128,128)),
            k=k
        )
        self.conv4 = DynamicEdgeConv(
            torch.nn.Sequential(torch.nn.Linear(256,256),
                                torch.nn.ReLU(),
                                torch.nn.Linear(256,256)),
            k=k
        )
        self.lin1 = torch.nn.Linear(512, emb_dims)
        self.bn1  = torch.nn.BatchNorm1d(emb_dims)
        self.dp1  = torch.nn.Dropout(dropout)
        self.lin2 = torch.nn.Linear(emb_dims, output_channels)

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
        return F.log_softmax(self.lin2(x), dim=1)


# -----------------------------------------------------------------------------
#  VALIDATION DATASET
# -----------------------------------------------------------------------------
class H5ValSet(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as hf:
            self.length = hf["data"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, "hf"):
            self.hf = h5py.File(self.path, "r")
        pts   = self.hf["data"][idx]
        label = int(self.hf["label"][idx])
        return Data(
            pos=torch.from_numpy(pts).float(),
            y=torch.tensor(label, dtype=torch.long),
        )


# -----------------------------------------------------------------------------
#  PLOTTING UTILITIES
# -----------------------------------------------------------------------------
def save_confusion_matrix(cm, epoch):
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Epoch {epoch}")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:d}",
                    ha="center",
                    color="white" if cm[i,j] > thresh else "black")
    out_path = os.path.join(EPOCHS_DIR, f"cm_epoch_{epoch:03d}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_roc_curves(y_true, y_score, epoch):
    y_bin    = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
    fpr = {}; tpr = {}; roc_auc = {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(8,8))
    plt.plot(fpr["macro"], tpr["macro"],
             label=f"Macro‑avg ROC (AUC = {roc_auc['macro']:.3f})",
             linewidth=2)
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Macro‑Average ROC — Epoch {epoch}")
    plt.legend(loc="lower right")
    out_path = os.path.join(EPOCHS_DIR, f"roc_epoch_{epoch:03d}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path, roc_auc["macro"]


# -----------------------------------------------------------------------------
#  EVALUATE ONE EPOCH
# -----------------------------------------------------------------------------
def evaluate_epoch(epoch, ckpt_path, val_loader, model):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_y, all_pred, all_prob = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch:03d}"):
            batch = batch.to(DEVICE)
            out   = model(batch)
            loss  = F.nll_loss(out, batch.y, reduction="sum")
            probs = out.exp()
            preds = probs.argmax(dim=1)

            total_loss += loss.item()
            all_y.append(batch.y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    loss_epoch = total_loss / len(val_loader.dataset)
    acc_epoch  = (y_pred == y_true).mean()
    f1_macro   = f1_score(y_true, y_pred, average="macro")
    auc_macro  = roc_auc_score(
        label_binarize(y_true, classes=np.arange(NUM_CLASSES)),
        y_prob,
        multi_class="ovr",
        average="macro"
    )

    cm   = confusion_matrix(y_true, y_pred)
    rpt  = classification_report(y_true, y_pred,
                                 labels=np.arange(NUM_CLASSES),
                                 output_dict=True)

    cm_path, = [save_confusion_matrix(cm, epoch)]
    roc_path, _ = save_roc_curves(y_true, y_prob, epoch)

    # dump JSON report
    import json
    with open(os.path.join(EPOCHS_DIR, f"report_epoch_{epoch:03d}.json"), "w") as f:
        json.dump(rpt, f, indent=2)

    return {
        "epoch":     epoch,
        "loss":      loss_epoch,
        "accuracy":  acc_epoch,
        "f1_macro":  f1_macro,
        "auc_macro": auc_macro,
    }


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------
def main():
    val_ds     = H5ValSet(VAL_H5)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = DGCNN().to(DEVICE)

    # gather checkpoints
    ckpts = []
    pat = re.compile(r"checkpoint_epoch_(\d+)\.pth$")
    for fn in os.listdir(CHECKPOINT_DIR):
        m = pat.match(fn)
        if m:
            ckpts.append((int(m.group(1)), os.path.join(CHECKPOINT_DIR, fn)))
    ckpts.sort(key=lambda x: x[0])

    results = []
    for epoch, ckpt_path in ckpts:
        res = evaluate_epoch(epoch, ckpt_path, val_loader, model)
        results.append(res)

    # global plots
    epochs = [r["epoch"] for r in results]
    for metric in ["loss", "accuracy", "f1_macro", "auc_macro"]:
        plt.figure()
        plt.plot(epochs, [r[metric] for r in results], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} vs. Epoch")
        plt.grid(True)
        plt.savefig(os.path.join(GLOBAL_DIR, f"{metric}_vs_epoch.png"),
                    bbox_inches="tight")
        plt.close()

    # CSV summary
    import csv
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["epoch", "loss", "accuracy", "f1_macro", "auc_macro"])
        for r in results:
            writer.writerow([r["epoch"], r["loss"], r["accuracy"],
                             r["f1_macro"], r["auc_macro"]])

    print(f"Done — all results in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
