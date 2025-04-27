import torch
import torch.nn as nn

# Générateur pour ModelNet40: génère un nuage de 1024 points en 3 dimensions
class generator_modelnet(nn.Module):
    def __init__(self, z_dim, num_points=1024):
        super(generator_modelnet, self).__init__()
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3),
            nn.Tanh()  # Suppose que les coordonnées sont dans [-1, 1]
        )
    
    def forward(self, z):
        x = self.fc(z)
        # Reshape pour obtenir [batch_size, num_points, 3]
        x = x.view(-1, self.num_points, 3)
        return x

# Discriminateur pour ModelNet40: juge la qualité d'un nuage de points (1024 points x 3 dimensions)
import torch.nn as nn

class discriminator_modelnet(nn.Module):
    def __init__(self, num_points=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: [batch, num_points, 3]
        x = x.transpose(1, 2)  # [batch, 3, num_points]
        x = self.conv(x)       # [batch, 256, num_points]
        x = torch.max(x, 2)[0] # [batch, 256]
        return self.fc(x)

