from data_utils import ChessDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models import ChessResNet
from data_utils import PerspectiveResEncoder
import torch.nn as nn
import torch.optim as optim
import torch
import os 

def training_and_validation(path_model, model : nn.Module, dataset : Dataset, device = "cpu", batch_size : int = 1024, train_size = 0.8, epochs = 10):
    
    train = int(train_size * len(dataset))
    test = len(dataset) - train
    trainset, testset = torch.utils.data.random_split(dataset, [train, test])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    net = model.to(device)

    if os.path.exists(path_model):
        print(f"Carico i pesi da '{path_model}'")
        # map_location=device serve a caricare i pesi direttamente sulla GPU
        net.load_state_dict(torch.load(path_model, map_location=device))
    else:
        print("Nessun fil etrovato, training da 0")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.0005, weight_decay=1e-5)

    best_test_loss = float('inf')
    patience = 3 # se dopo 3 epoche non miglioro, mi fermo
    trigger_times = 0 # numero di epoche consecutive senza miglioramento

    for epoch in range(epochs):
        
        # Traing
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
            
        # Validation
        net.eval()  # Mette la rete in modalità valutazione
        test_loss = 0.0
        with torch.no_grad():  
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                outputs = net(inputs)
                loss_val = criterion(outputs.view(-1), labels.view(-1))
                test_loss += loss_val.item()
                
        avg_test_loss = test_loss / len(testloader)
        print(f'Fine Epoca {epoch + 1} - Average Test Loss: {test_loss / len(testloader):.3f}')
            
        # Elary Stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(net.state_dict(), path_model)
            trigger_times = 0
            print("Pesi aggiornati")
        else:
            trigger_times += 1
            print(f"Nessuno miglioramento per {trigger_times} epoche")
            if trigger_times >= patience:
                print("Early Stopping")
                break

    print('Finished Training')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    path_model = "chess_model2.pth"
    model = ChessResNet()
    encoder = PerspectiveResEncoder()
    dataset = ChessDataset("../data/chessData.csv", encoder=encoder)

    training_and_validation(path_model, model, dataset, device=device)

