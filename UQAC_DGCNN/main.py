import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, output_channels=40):
        """
        k           : number of neighbors in k-NN used by DynamicEdgeConv.
        emb_dims    : embedding dimension.
        dropout     : dropout rate.
        output_channels : number of classes (40 for ModelNet40).
        """
        super(DGCNN, self).__init__()
        self.k = k

        # First layer: input = points (3 dims), so 2*3 = 6 dimensions.
        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            ),
            k=k
        )
        # Second layer: input features from previous layer (64 dims) => 2*64 = 128.
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            ),
            k=k
        )
        # Third layer.
        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            ),
            k=k
        )
        # Fourth layer.
        self.conv4 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ),
            k=k
        )

        # After concatenating the outputs (64 + 64 + 128 + 256 = 512)
        self.lin1 = nn.Linear(512, emb_dims)
        self.bn1 = nn.BatchNorm1d(emb_dims)
        self.dp1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(emb_dims, output_channels)

    def forward(self, data):
        # Use positions (data.pos) as input.
        x, batch = data.pos, data.batch  # x shape: [N, 3]
        
        x1 = self.conv1(x, batch)  # [N, 64]
        x2 = self.conv2(x1, batch)  # [N, 64]
        x3 = self.conv3(x2, batch)  # [N, 128]
        x4 = self.conv4(x3, batch)  # [N, 256]

        # Concatenate outputs from the 4 layers -> [N, 512]
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)

        # Global max pooling per graph.
        x_pool = global_max_pool(x_cat, batch)

        # Final classification MLP.
        x = self.lin1(x_pool)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            data = data.to(device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def plot_metrics(train_losses, val_losses, val_accuracies, epoch, save_dir='plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(10,4))
    
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(range(5, epoch+1, 5), val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epoch")
    
    plt.subplot(1,2,2)
    plt.plot(range(5, epoch+1, 5), val_accuracies, label="Val Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy vs Epoch")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    plt.close()

def main():
    # Use CUDA if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Compose transforms: normalize scale and sample a fixed number of points.
    transform = T.Compose([
        T.NormalizeScale(),
        T.SamplePoints(1024)
    ])

    # Download and load the full ModelNet40 dataset.
    dataset = ModelNet(root='./data/ModelNet40', name='40', transform=transform)
    dataset = dataset.shuffle()

    # Split the dataset into training and testing sets (using provided splits).
    # ModelNet40 provides standard train/test splits, but here we use a simple 80/20 split.
    split = int(0.8 * len(dataset))
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]
    print(f"Training on {len(train_dataset)} samples; validating on {len(test_dataset)} samples.")

    # Use batch size 32 and a modest number of workers.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    # Initialize model for 40 classes.
    model = DGCNN(output_channels=40).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    best_val_accuracy = 0
    best_epoch = 0
    patience = 5   # Number of validation steps to wait for improvement
    no_improve_count = 0

    num_epochs =  10000  # Maximum epochs; early stopping may terminate before this

    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, optimizer, train_loader, device)
        train_loss_history.append(train_loss)
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            val_loss, val_accuracy = evaluate(model, test_loader, device)
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)
            print(f"  [Validation] Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Plot metrics at this validation step
            plot_metrics(train_loss_history, val_loss_history, val_accuracy_history, epoch)
            
            # Check for improvement
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                no_improve_count = 0
                # Save the best model
                torch.save(model.state_dict(), "dgcnn_model_best.pth")
                print(f"  [Validation] New best model saved at epoch {epoch}.")
            else:
                no_improve_count += 1
                print(f"  [Validation] No improvement count: {no_improve_count}/{patience}.")
            
            # Early stopping if no improvement for 'patience' validations
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}.")
                break

    print(f'Best Validation Accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}.')
    print("Best model saved as 'dgcnn_model_best.pth'.")

if __name__ == "__main__":
    main()
