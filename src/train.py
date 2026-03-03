import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import subprocess

# Import dal tuo progetto
from models import DeepChessNet, ChessResNet # Ho aggiunto entrambi per sicurezza
from data_utils import ChessDataset, PerspectiveResEncoder

def git_push_weights(filepath, loss_val, epoch):
    """
    Funzione per committare e pushare automaticamente i pesi su GitHub.
    """
    try:
        subprocess.run(["git", "add", filepath], check=True)
        message = f"Update weights Epoch {epoch+1} - Loss: {loss_val:.5f}"
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"Pesi inviati a GitHub con successo.")
    except Exception as e:
        print(f"Errore durante il Git Push: {e}")

def training_and_validation(path_model, model: nn.Module, dataset: Dataset, device="cpu", 
                            batch_size: int = 1024, train_size=0.9, epochs=50):
    
    train_len = int(train_size * len(dataset))
    test_len = len(dataset) - train_len
    trainset, testset = random_split(dataset, [train_len, test_len])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)

    net = model.to(device)

    # Caricamento pesi esistenti
    if os.path.exists(path_model):
        print(f"Carico i pesi esistenti da '{path_model}'")
        net.load_state_dict(torch.load(path_model, map_location=device))
    else:
        print(f"Nessun file trovato in {path_model}, inizio training da zero.")

    criterion = nn.MSELoss()
    # AdamW è ottimo per le ResNet profonde
    optimizer = optim.AdamW(net.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # Scheduler: riduce il LR se la loss smette di scendere
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_test_loss = float('inf')
    early_stop_patience = 5
    trigger_times = 0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_val = criterion(outputs, labels)
            
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] Loss: {running_loss / 100:.5f}')
                running_loss = 0.0
            
        # Validation Phase
        net.eval()
        test_loss = 0.0
        with torch.no_grad():  
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                outputs = net(inputs)
                loss_val = criterion(outputs, labels)
                test_loss += loss_val.item()
                
        avg_test_loss = test_loss / len(testloader)
        print(f' Fine Epoca {epoch + 1} - Average Test Loss: {avg_test_loss:.5f}')
            
        # Scheduler Step
        scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(net.state_dict(), path_model)
            trigger_times = 0
            print(f" Avarage Loss Test migliorata. Pesi salvati in {path_model}.")
            
            # git psuh 
            git_push_weights(path_model, avg_test_loss, epoch)
        else:
            trigger_times += 1
            print(f"Nessun miglioramento per {trigger_times} epoche")
            if trigger_times >= early_stop_patience:
                print("Early Stopping attivato.")
                break

    print('Training Completato.')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    path_model = "src/chess_model3.pth" 
    
    model = DeepChessNet(num_blocks=20) 
    
    encoder = PerspectiveResEncoder()
    
    # Percorso del dataset su Kaggle
    dataset_path = "/kaggle/input/datasets/ronakbadhe/chess-evaluations/chessData.csv"
    
    if os.path.exists(dataset_path):
        dataset = ChessDataset(dataset_path, encoder=encoder)
        training_and_validation(path_model, model, dataset, device=device, batch_size=1024, epochs=50)
    else:
        print(f"Errore: Dataset non trovato in {dataset_path}")