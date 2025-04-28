import os
import random  # Import nécessaire pour mélanger la liste
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

# Pour les métriques additionnelles
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ------------------------------
# Définition du modèle DGCNN
# ------------------------------
class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, output_channels=40):
        """
        k           : nombre de voisins utilisés par DynamicEdgeConv.
        emb_dims    : dimension de l'embedding.
        dropout     : taux de dropout.
        output_channels : nombre de classes (40 pour ModelNet40).
        """
        super(DGCNN, self).__init__()
        self.k = k

        # Couche 1 : entrée = points (3 dims) => 2*3 = 6 dims.
        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            ),
            k=k
        )
        # Couche 2
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            ),
            k=k
        )
        # Couche 3
        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            ),
            k=k
        )
        # Couche 4
        self.conv4 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ),
            k=k
        )

        # Après concaténation (64 + 64 + 128 + 256 = 512)
        self.lin1 = nn.Linear(512, emb_dims)
        self.bn1 = nn.BatchNorm1d(emb_dims)
        self.dp1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(emb_dims, output_channels)

    def forward(self, data):
        # Utilisation des positions (data.pos) comme entrée.
        x, batch = data.pos, data.batch  # x shape: [N, 3]
        
        x1 = self.conv1(x, batch)  # [N, 64]
        x2 = self.conv2(x1, batch)  # [N, 64]
        x3 = self.conv3(x2, batch)  # [N, 128]
        x4 = self.conv4(x3, batch)  # [N, 256]

        # Concaténation des sorties (64+64+128+256 = 512)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        # Pooling global sur chaque graphe
        x_pool = global_max_pool(x_cat, batch)
        # Classification finale par MLP.
        x = self.lin1(x_pool)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

# ------------------------------
# Boucle d'entraînement et évaluation de base
# ------------------------------
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
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            data = data.to(device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            # Conserver pour les métriques : on convertit le log_softmax en probabilités
            prob = torch.exp(out)
            all_preds.append(pred.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy, np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)

# ------------------------------
# Visualisation des métriques d'entraînement et des scores ROC
# ------------------------------
def plot_metrics(train_losses, val_losses, val_accuracies, epoch, save_dir='plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(10,4))
    
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(range(3, epoch+1, 3), val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epoch")
    
    plt.subplot(1,2,2)
    plt.plot(range(3, epoch+1, 3), val_accuracies, label="Val Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy vs Epoch")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    plt.close()

def plot_roc_curves(y_true, y_score, n_classes, save_path='plots/roc_curve.png'):
    # Binariser les labels
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    # Calcul de la micro-average ROC curve et AUC
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f"micro-average ROC curve (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

# ------------------------------
# Visualisation des nuages de points pour quelques exemples
# ------------------------------
def visualize_point_clouds(model, loader, device, class_names, save_path='plots/point_clouds.png'):
    model.eval()
    correct_samples = []
    wrong_samples = []
    # On itère sur le loader (en batch) pour récupérer chaque échantillon individuellement
    with torch.no_grad():
        for data in tqdm(loader, desc="Collecting Samples for Visualization", leave=False):
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            # On extrait chaque exemple du batch en se basant sur data.batch
            for i in range(data.num_graphs):
                idx = (data.batch == i)
                pts = data.pos[idx].cpu().numpy()
                true_label = int(data.y[i].cpu().numpy())
                predicted_label = int(pred[i].cpu().numpy())
                sample_info = {
                    'points': pts,
                    'true': true_label,
                    'pred': predicted_label
                }
                if true_label == predicted_label and len(correct_samples) < 5:
                    correct_samples.append(sample_info)
                elif true_label != predicted_label and len(wrong_samples) < 5:
                    wrong_samples.append(sample_info)
                if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
                    break
            if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
                break

    # Visualisation des échantillons avec matplotlib en 3D
    total = len(correct_samples) + len(wrong_samples)
    fig = plt.figure(figsize=(15, 3 * total))
    all_samples = correct_samples + wrong_samples
    titles = ["Correct"] * len(correct_samples) + ["Incorrect"] * len(wrong_samples)
    for idx, sample in enumerate(all_samples):
        ax = fig.add_subplot(total, 1, idx+1, projection='3d')
        pts = sample['points']
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
        ax.set_title(f"{titles[idx]} sample - GT: {class_names[sample['true']]} / Pred: {class_names[sample['pred']]}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ------------------------------
# Fonction principale
# ------------------------------
def main():
    # Utilisation du GPU si disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transformations sur les données
    transform = T.Compose([
        T.NormalizeScale(),
    ])

    # Chargement du dataset ModelNet40
    dataset = ModelNet(root='./data/ModelNet40', name='40', transform=transform)
    print(f"Dataset loaded with {len(dataset)} samples.")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of features: {dataset.num_features}")
    print('Preview of the dataset:')
    print(dataset[0])
    
    filtered_data = [data for data in dataset if data.pos.size(0) >= 2048]

    # Application de la transformation T.SamplePoints(1024)
    transform = T.Compose([
        T.SamplePoints(1024)  # On échantillonne à 1024 points
    ])
    filtered_data = [transform(data) for data in filtered_data]

    print(f"Filtered dataset size: {len(filtered_data)} samples with at least 2048 points.")
    print("Preview of the filtered dataset:")
    print(filtered_data[0])

    # Mélanger les données avec random.shuffle()
    random.shuffle(filtered_data)
    dataset = filtered_data
    split = int(0.8 * len(dataset))
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]
    print(f"Training on {len(train_dataset)} samples; validating on {len(test_dataset)} samples.")

    # Création des DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    # Liste des classes (ModelNet40 contient 40 classes, ici on suppose que dataset.classes existe sinon adaptez)
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    else:
        class_names = [str(i) for i in range(40)]

    # Initialisation du modèle et de l'optimiseur
    model = DGCNN(output_channels=40).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Boucle d'entraînement avec early stopping
    best_val_accuracy = 0
    best_epoch = 0
    patience = 2   # patience en validations consécutives sans amélioration
    no_improve_count = 0
    num_epochs = 10000

    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, optimizer, train_loader, device)
        train_loss_history.append(train_loss)
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")
        
        if epoch % 3 == 0:
            val_loss, val_accuracy, _, _, _ = evaluate(model, test_loader, device)
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)
            print(f"  [Validation] Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            plot_metrics(train_loss_history, val_loss_history, val_accuracy_history, epoch)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                no_improve_count = 0
                torch.save(model.state_dict(), "dgcnn_model_best.pth")
                print(f"  [Validation] New best model saved at epoch {epoch}.")
            else:
                no_improve_count += 1
                print(f"  [Validation] No improvement count: {no_improve_count}/{patience}.")
            
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}.")
                break

    print(f'Best Validation Accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}.')
    print("Best model saved as 'dgcnn_model_best.pth'.")

    # ------------------------------
    # Évaluation détaillée sur l'ensemble de test
    # ------------------------------
    val_loss, val_accuracy, y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Score F1 (moyenne macro)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Score (macro): {f1:.4f}")

    # ROC-AUC pour classification multi-classes (micro-average)
    y_true_bin = label_binarize(y_true, classes=np.arange(40))
    roc_auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
    print(f"ROC-AUC Score (macro, OVR): {roc_auc:.4f}")
    plot_roc_curves(y_true, y_prob, n_classes=40, save_path='plots/roc_curve.png')
    print("ROC curve saved as 'plots/roc_curve.png'.")

    # ------------------------------
    # Visualisation des nuages de points pour quelques exemples
    # ------------------------------
    visualize_point_clouds(model, test_loader, device, class_names, save_path='plots/point_clouds.png')
    print("Point clouds visualization saved as 'plots/point_clouds.png'.")

if __name__ == "__main__":
    main()
