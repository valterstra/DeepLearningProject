import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import numpy as np
import random
import matplotlib.pyplot as plt


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(batch_size=64, val_size=1000, use_random_crop=False, use_random_flip=False, use_normalization=False):
    transform_list = []

    if use_random_crop:
        transform_list.append(transforms.RandomCrop(32, padding=4))
    if use_random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())

    if use_normalization:
        transform_list.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )

    transform = transforms.Compose(transform_list)

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]) if use_normalization else transforms.ToTensor()

    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=base_transform)

    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class CIFAR10Model(nn.Module):
    def __init__(self, use_dropout=True, dropout_rate=0.2):
        super().__init__()
        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()

        # Block 1: 32x32 → 16x16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 16x16 → 8x8
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 8x8 → 4x4
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

        # He initialization
        for layer in [
            self.conv1_1, self.conv1_2,
            self.conv2_1, self.conv2_2,
            self.conv3_1, self.conv3_2,
            self.fc1, self.fc2
        ]:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu' if layer != self.fc2 else 'linear')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total += y.size(0)
        correct += (preds.argmax(1) == y).sum().item()

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)
            total += y.size(0)
            correct += (preds.argmax(1) == y).sum().item()

    return total_loss / total, correct / total


def plot_diagnostics(train_acc, val_acc, train_loss, val_loss):
    epochs = range(1, len(train_acc) + 1)

    plt.figure()
    plt.plot(epochs, train_acc, label='Train acc')
    plt.plot(epochs, val_acc, label='Val acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_loss, label='Train loss')
    plt.plot(epochs, val_loss, label='Val loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # augmentations
    use_random_crop = True #this is the translation agumentation that they are talking about
    use_random_flip = True
    use_normalization = True

    use_dropout = False
    dropout_rate = 0

    train_loader, val_loader, test_loader = load_data(
        batch_size=64,
        val_size=1000,
        use_random_crop=use_random_crop,
        use_random_flip=use_random_flip,
        use_normalization=use_normalization
    )

    model = CIFAR10Model(use_dropout=use_dropout, dropout_rate=dropout_rate).to(device)


    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=0,
        amsgrad=False
    )
    criterion = nn.CrossEntropyLoss()

    epochs = 100
    train_acc, val_acc = [], []
    train_loss, val_loss = [], []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        val_loss.append(va_loss)
        val_acc.append(va_acc)

        print(f'Epoch {epoch}/{epochs} — '
              f'Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} — '
              f'Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}')

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'> Final test accuracy: {test_acc * 100:.2f}%')

    plot_diagnostics(train_acc, val_acc, train_loss, val_loss)


if __name__ == '__main__':
    main()
