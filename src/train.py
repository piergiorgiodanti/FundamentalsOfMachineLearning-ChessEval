import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import gc

from models import ChessVanillaCNN, ChessResNet, DeepChessResNet, TransformerChessEval
from data_utils import ChessDataset, PerspectiveVectorizer, StaticFlatVectorizer

STEPS_PRINT = 100
BATCH_SIZE = 2048
EPOCHS = 10
DATA_PATH = "../data/chessData.csv"

def training_loop(path_model, model, train_set, test_set, device, lr):
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    net = model.to(device)

    if os.path.exists(path_model):
        print(f"Ripristino pesi da: {path_model}")
        net.load_state_dict(torch.load(path_model, map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)

    scaler = torch.amp.GradScaler('cuda')

    best_test_loss = float('inf')
    patience = 3
    trigger_times = 0

    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if i % STEPS_PRINT == (STEPS_PRINT - 1):
                print(f'[Epoca {epoch + 1}, Batch {i + 1}] Loss: {running_loss / STEPS_PRINT:.4f}')
                running_loss = 0.0

        # Validation
        net.eval()
        test_loss = 0.0
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                outputs = net(inputs)
                test_loss += criterion(outputs, labels).item()

        avg_loss = test_loss / len(test_loader)
        print(f'>>> Fine Epoca {epoch + 1} - Val Loss: {avg_loss:.4f}')

        if avg_loss < best_test_loss:
            best_test_loss = avg_loss
            torch.save(net.state_dict(), path_model)
            print("Modello salvato.")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early Stopping.")
                break

    # Pulizia memoria per prossimo modello
    del net, optimizer, train_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione su: {device}")

    print("Caricamento dataset...")

    full_dataset = ChessDataset(DATA_PATH, vectorizer=None)

    train_len = int(0.8 * len(full_dataset))
    test_len = len(full_dataset) - train_len

    generator = torch.Generator().manual_seed(42)
    train_set, test_set = random_split(full_dataset, [train_len, test_len], generator=generator)

    configs = [
        (ChessVanillaCNN(), "../models/vanilla_cnn.pth", 0.0005, StaticFlatVectorizer()),
        (ChessResNet(num_blocks=8), "../models/resnet_8.pth", 0.0005, PerspectiveVectorizer()),
        (DeepChessResNet(num_blocks=20), "../models/resnet_20.pth", 0.0004, PerspectiveVectorizer()),
        (TransformerChessEval(), "../models/transformer_encoder.pth", 0.0001, PerspectiveVectorizer())
    ]

    for model, filename, lr, vectorizer in configs:
        print(f" TRAINING: {model.__class__.__name__} ")
        print(f" Vectorizer in uso: {vectorizer.__class__.__name__} ({vectorizer.layers} canali)")

        full_dataset.vectorizer = vectorizer

        training_loop(filename, model, train_set, test_set, device, lr)

if __name__ == '__main__':
    main()