from DCGANpytorch import generator_modelnet, discriminator_modelnet
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Parameters
batch_size = 32  # Increased batch size
image_size = 1024
z_dim = 200
lr_g = 0.0002  # Generator learning rate
lr_d = 0.0004  # Discriminator learning rate (slightly higher)
beta1 = 0.5
beta2 = 0.999
epochs = 5000  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Use CUDA if available.
    print(f"Using device: {device}")

    # Compose transforms: normalize scale and sample a fixed number of points.
    transform = T.Compose([
        T.NormalizeScale(),
        T.SamplePoints(1024)
    ])

    # Download and load the full ModelNet40 dataset.
    dataset = ModelNet(root='./data/ModelNet40', name='40', transform=transform)

    print(dataset)
    dataset = dataset.shuffle()
    
    target_index = 6  # Indice de la classe "bowl" dans ModelNet40
    bowl_dataset = [data for data in dataset if data.y.item() == target_index]

    print(f"Nombre d'exemples 'bowl': {len(bowl_dataset)}")

    # Séparation du dataset en train et test
    split = int(0.8 * len(dataset))
    full_train = dataset[:split]
    full_test = dataset[split:]

    # Filtrer pour ne conserver que les exemples de la classe "bowl"
    train_dataset = [data for data in full_train if data.y.item() == target_index]
    test_dataset = [data for data in full_test if data.y.item() == target_index]

    print(f"Training on {len(train_dataset)} examples (classe bowl); validating on {len(test_dataset)} examples (classe bowl).")

    # Use batch size 32 and a modest number of workers.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    # ---------------------- instanciation des modèles ------------------
    netG = generator_modelnet(z_dim, image_size).to(device)
    netD = discriminator_modelnet(image_size).to(device)

    # ---------------------- initialisation des poids et autres états -------------------
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))

    dataloader = train_loader
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            # real_data: [batch_size, num_points, 3]
            current_batch = batch.num_graphs  # nombre d'exemples dans le batch
            real_data = batch.pos.view(current_batch, image_size, 3).to(device)
            
            # ===== Train Discriminator =====
            netD.zero_grad()
            
            # Train with real data (étiquettes légèrement lissées)
            label_real = torch.full((current_batch, 1), 0.9, device=device)
            output_real = netD(real_data)
            lossD_real = criterion(output_real, label_real)
            
            # Ajout d'un peu de bruit aux nuages réels
            real_data_noisy = real_data + 0.05 * torch.randn_like(real_data)
            output_real_noisy = netD(real_data_noisy)
            lossD_real_noisy = criterion(output_real_noisy, label_real)
            
            # Train with fake data
            noise = torch.randn(current_batch, z_dim, device=device)
            if random.random() > 0.5:
                noise = noise + 0.1 * torch.randn_like(noise)
            
            fake_data = netG(noise)
            label_fake = torch.full((current_batch, 1), 0.0, device=device)
            output_fake = netD(fake_data.detach())
            lossD_fake = criterion(output_fake, label_fake)
            
            lossD = lossD_real + lossD_fake + 0.5 * lossD_real_noisy
            lossD.backward()
            optimizerD.step()
            
            # ===== Train Generator =====
            netG.zero_grad()
            noise = torch.randn(current_batch, z_dim, device=device)
            fake_data = netG(noise)
            label_gen = torch.full((current_batch, 1), 1.0, device=device)
            output_gen = netD(fake_data)
            
            lossG = criterion(output_gen, label_gen)
            lossG.backward()
            optimizerG.step()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {i}/{len(dataloader)} | LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")

    # Enregistrement des modèles
    torch.save(netG.state_dict(), 'generator_modelnet.pth')
    torch.save(netD.state_dict(), 'discriminator_modelnet.pth')

    # Génération et affichage de nuages de points générés
    n_samples = 4
    noise = torch.randn(n_samples, z_dim, device=device)
    with torch.no_grad():
        fake_points = netG(noise)
    fake_points = fake_points.cpu().numpy()

    fig = plt.figure(figsize=(8,8))
    for i in range(n_samples):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        pc = fake_points[i]
        ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=1, c='b')
        ax.set_title(f"Nuage {i+1}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()