# ! pip install torch torch_geometric torch_cluster torch_scatter torch_sparse tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool, knn_graph
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import random
from tqdm import tqdm

# --------- Efficient EdgeConv using MessagePassing ---------
class EfficientEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, k=5, aggr='max'):
        super().__init__(aggr=aggr)
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, batch):
        # Compute the k-NN graph; this returns a tensor [2, num_edges]
        edge_index = knn_graph(x, k=self.k, batch=batch)
        # Propagate messages – this will call self.message for each edge and aggregate them.
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        # Compute the message using neighbor differences and neighbor features.
        # The operation is computed in a streaming fashion, reducing memory usage.
        message_input = torch.cat([x_i - x_j, x_j], dim=1)
        return self.mlp(message_input)

# --------- DGCNN Model using EfficientEdgeConv ---------
class DGCNN(nn.Module):
    def __init__(self, k=5, emb_dims=1024):
        super().__init__()
        self.ec1 = EfficientEdgeConv(3, 64, k)
        self.ec2 = EfficientEdgeConv(64, 64, k)
        self.ec3 = EfficientEdgeConv(64, 128, k)
        self.ec4 = EfficientEdgeConv(128, emb_dims, k)
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
transform = Compose([NormalizeScale(), SamplePoints(1024/4)])
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

# Use DataLoader (you may adjust batch size/workers as needed)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=transform_collate)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=transform_collate)

# --------- Main Training Loop with Mixed Precision and TQDM ---------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision setup (using updated syntax)
    scaler = torch.amp.GradScaler(device='cuda')
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=True)
        for data in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            
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
        torch.cuda.empty_cache()
    
    print("Training complete.")
    torch.cuda.empty_cache()