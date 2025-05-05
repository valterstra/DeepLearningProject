import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CIFAR10WithOneHot(CIFAR10):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label

def prepare_data_loaders(batch_size=100, val_size=1000):
    transform = transforms.ToTensor()

    full_train = CIFAR10WithOneHot(root='.', train=True, download=True, transform=transform)
    test_set = CIFAR10WithOneHot(root='.', train=False, download=True, transform=transform)

    train_size = len(full_train) - val_size
    train_set, val_set = torch.utils.data.random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader

class ConvNetBaseline(nn.Module):
    """original patchify → ReLU → FC → FC"""
    def __init__(self, f=4, nf=3, nh=50, channels=3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(channels, nf, kernel_size=f, stride=f)
        self.fc1 = nn.Linear(nf * (32 // f) * (32 // f), nh)
        self.fc2 = nn.Linear(nh, num_classes)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNetVGG1Block(nn.Module):
    """patchify → VGG-style block → FC → FC"""
    def __init__(self, nf=64, nh=128, channels=3, num_classes=10):
        super().__init__()
        self.initial_conv = nn.Conv2d(channels, nf, kernel_size=3, stride=1, padding=1)
        self.vgg_block = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        side = 32 // 2
        self.fc1 = nn.Linear(nf * side * side, nh)
        self.fc2 = nn.Linear(nh, num_classes)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        x = self.vgg_block(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvNetVGG3Blocks(nn.Module):
    """Patchify + 3 VGG-style blocks (no final pool) → FC → FC, optional Dropout"""
    def __init__(self, nf=64, nh=128, channels=3, num_classes=10, use_dropout=False, p_drop=0.5):
        super().__init__()
        self.use_dropout = use_dropout

        self.initial_conv = nn.Conv2d(channels, nf, kernel_size=3, stride=1, padding=1)
        if self.use_dropout:
            self.dropout0 = nn.Dropout(p=p_drop)

        self.block1 = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        if self.use_dropout:
            self.dropout1 = nn.Dropout(p=p_drop)

        self.block2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        if self.use_dropout:
            self.dropout2 = nn.Dropout(p=p_drop)

        self.block3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(nf * 4, nf * 4, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        if self.use_dropout:
            self.dropout3 = nn.Dropout(p=p_drop)

        side = 32 // 4
        self.fc1 = nn.Linear(nf * 4 * side * side, nh)
        if self.use_dropout:
            self.dropout4 = nn.Dropout(p=p_drop)
        self.fc2 = nn.Linear(nh, num_classes)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        if self.use_dropout:
            x = self.dropout0(x)
        x = self.block1(x)
        if self.use_dropout:
            x = self.dropout1(x)
        x = self.block2(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.block3(x)
        if self.use_dropout:
            x = self.dropout3(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout4(x)
        x = self.fc2(x)
        return x


def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            preds = model(data).argmax(dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)
    return correct / total


def train_loop(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    return running_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, n_epochs):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        train_loss = train_loop(model, train_loader, optimizer, criterion, device)
        train_acc = compute_accuracy(model, train_loader, device)

        model.eval()
        val_loss = 0.0
        correct = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            val_loss += criterion(outputs, target).item() * data.size(0)
            correct += (outputs.argmax(dim=1) == target).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc  = correct / len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"epoch {epoch:02d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    total_time = time.time() - start_time
    return history, total_time


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # architecture selection: 0=baseline, 1=1-block, 3=3-blocks
    arch = 3  # set to 0, 1, or 3
    use_dropout = True  # added boolean for dropout
    p_drop = 0.3

    batch_size = 100
    n_epochs = 80

    if arch == 0:
        f, nf, nh = 4, 3, 50
        lr, wd = 0.01, 0.003
    else:
        f, nf, nh = 2, 64, 128
        lr, wd = 1e-3, None

    train_loader, val_loader, test_loader = prepare_data_loaders(batch_size=batch_size)

    if arch == 0:
        model = ConvNetBaseline(nf=nf, nh=nh).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    elif arch == 1:
        model = ConvNetVGG1Block(nf=nf, nh=nh).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        model = ConvNetVGG3Blocks(nf=nf, nh=nh, use_dropout=use_dropout, p_drop=p_drop).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    history, total_time = train_model(model, train_loader, val_loader,
                                      optimizer, criterion, device, n_epochs)
    print(f"training completed in {total_time:.2f} seconds")
    test_acc = compute_accuracy(model, test_loader, device)
    print(f"test accuracy: {test_acc:.4f}")

    epochs = range(1, n_epochs + 1)
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()