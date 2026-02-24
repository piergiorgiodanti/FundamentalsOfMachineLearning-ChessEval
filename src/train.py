from data_utils import ChessDataset
from models import ChessEval
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import os 

path_modello = "chess_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"device: {device} ({torch.cuda.get_device_name(0)})")
batch_size = 1024

dataset = ChessDataset("../data/chessData.csv")

# 80% train, 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

net = ChessEval().to(device)

if os.path.exists(path_modello):
    print(f"Carico i pesi da '{path_modello}'")
    # map_location=device serve a caricare i pesi direttamente sulla GPU
    net.load_state_dict(torch.load(path_modello, map_location=device))
else:
    print("Nessun fil etrovato, training da 0")

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(1):
    
    running_loss = 0.0
    net.train()
    
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        optimizer.zero_grad()

        outputs = net(inputs)

        loss_val = criterion(outputs.view(-1), labels.view(-1))

        loss_val.backward()
        optimizer.step()

        running_loss += loss_val.item()
        if i % 100 == 99:    # stampa ogni 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
        
        if i % 500 == 499:
            # salva i pesi del modello
            torch.save(net.state_dict(), "chess_model.pth")

    # --- FASE DI TESTING (VALIDATION) ---
    net.eval()  # Mette la rete in modalità valutazione
    test_loss = 0.0
    with torch.no_grad():  
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            outputs = net(inputs)
            loss_val = criterion(outputs.view(-1), labels)
            test_loss += loss_val.item()
            
    print(f'Fine Epoca {epoch + 1} - Average Test Loss: {test_loss / len(testloader):.3f}')
        
print('Finished Training')