import torch.nn as nn
import torch.nn.functional as F
import torch

class ChessEval(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh() # per schiacciare l'output in [-1,1]
        )
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
        
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x # Salvo l'originale
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # Trasformazione + Originale
        return F.relu(out)

class ChessResNet(nn.Module):
    def __init__(self, num_blocks=8):
        super().__init__()
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(17, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.res_tower = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_blocks)]
        )

        self.reduction = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(2048, 512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.res_tower(x)
        x = self.reduction(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        return torch.tanh(self.fc2(x))