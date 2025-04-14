# ! pip install torch torch_geometric torch_cluster torch_scatter torch_sparse tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
from torch_geometric.nn import knn_graph, global_max_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import random
from tqdm import tqdm  # Import tqdm for progress bars

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
        # Create a tensor of shape [num_nodes, out_channels] with same dtype as out
        out_final = torch.zeros(x.size(0), out.size(1), device=x.device, dtype=out.dtype)
        out_final = out_final.scatter_add_(0, row.unsqueeze(-1).expand_as(out), out)
        return out_final

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

# --------- Transforms and DataLoader Setup ---------
transform = Compose([NormalizeScale(), SamplePoints(1024)])
full_dataset = ModelNet(root='./data/ModelNet40', name='40', train=True, pre_transform=None)

# Randomly sample 300 examples
subset_indices = random.sample(range(len(full_dataset)), 300)
small_dataset = Subset(full_dataset, subset_indices)

def transform_collate(batch):
    batch = [transform(data) for data in batch]
    return torch.utils.data.default_collate(batch)

# Split the dataset into training and testing sets
split = int(0.8 * len(small_dataset))
train_dataset = small_dataset[:split]
test_dataset = small_dataset[split:]
print(f"Training on {len(train_dataset)} samples; validating on {len(test_dataset)} samples.")

# Use DataLoader with reduced batch size and fewer workers to ease memory usage
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=transform_collate)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=transform_collate)

# --------- Main Training Loop ---------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Updated mixed precision setup
    scaler = torch.amp.GradScaler(device='cuda')
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # Create a tqdm progress bar that updates every step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=True)
        
        for data in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Use autocast with the new recommended syntax
            with torch.amp.autocast(device_type='cuda'):
                out = model(data)
                loss = criterion(out, data.y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Clear cached memory to reduce fragmentation
        torch.cuda.empty_cache()
    
    print("Training complete.")
