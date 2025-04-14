# ! pip install torch torch_geometric torch_cluster torch_scatter torch_sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
from torch_geometric.nn import knn_graph, global_max_pool
from torch_geometric.loader import DataLoader

# --------- EdgeConv ---------
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, batch):
        edge_index = knn_graph(x, k=self.k, batch=batch)
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        edge_features = torch.cat([x_i - x_j, x_j], dim=1)
        out = self.mlp(edge_features)
        out = torch.zeros_like(x).scatter_add_(0, row.unsqueeze(-1).expand_as(out), out)
        return out

# --------- DGCNN Model ---------
class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024):
        super().__init__()
        self.ec1 = EdgeConv(3, 64, k)
        self.ec2 = EdgeConv(64, 64, k)
        self.ec3 = EdgeConv(64, 128, k)
        self.ec4 = EdgeConv(128, emb_dims, k)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dims, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 40)
        )

    def forward(self, data):
        x, batch = data.pos, data.batch
        x = self.ec1(x, batch)
        x = self.ec2(x, batch)
        x = self.ec3(x, batch)
        x = self.ec4(x, batch)
        x = global_max_pool(x, batch)
        return self.classifier(x)

# --------- Transforms ---------

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import NormalizeScale, SamplePoints, Compose
from torch.utils.data import DataLoader, Subset
import random

transform = Compose([NormalizeScale(), SamplePoints(1024)])
full_dataset = ModelNet(root='ModelNet40', name='40', train=True, pre_transform=None)

# Garde aléatoirement 300 exemples
subset_indices = random.sample(range(len(full_dataset)), 300)
small_dataset = Subset(full_dataset, subset_indices)

def transform_collate(batch):
    batch = [transform(data) for data in batch]
    return torch.utils.data.default_collate(batch)

# train_loader = DataLoader(small_dataset, batch_size=16, shuffle=True, collate_fn=transform_collate)


# # Compose transforms: normalize scale and sample a fixed number of points.
# transform = Compose([
#     NormalizeScale(),
#     SamplePoints(1024)
# ])

# # Download and load the full ModelNet40 dataset.
# dataset = ModelNet(root='ModelNet40', name='40', transform=transform)
# dataset = dataset.shuffle()

# Split the dataset into training and testing sets (using provided splits).
# ModelNet40 provides standard train/test splits, but here we use a simple 80/20 split.
split = int(0.8 * len(small_dataset))
train_dataset = small_dataset[:split]
test_dataset = small_dataset[split:]
print(f"Training on {len(train_dataset)} samples; validating on {len(test_dataset)} samples.")

# Use batch size 32 and a modest number of workers.
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1, collate_fn=transform_collate)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1, collate_fn=transform_collate)

# --------- Entraînement ---------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

