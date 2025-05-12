import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms, datasets
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


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

class CIFAR10Noisy(datasets.CIFAR10):
    """
    CIFAR-10 dataset with optional symmetric or asymmetric label noise.
    """
    def __init__(self, root, train=True, transform=None, download=False,
                 noise_rate=0.0, asym=False, seed=0):
        super().__init__(root, train=train, transform=transform, download=download)
        np.random.seed(seed)
        self.noise_rate = noise_rate
        self.asym = asym
        self._inject_noise()

    def _inject_noise(self):
        if self.asym:
            source = [9, 2, 3, 5, 4]
            target = [1, 0, 5, 3, 7]
            for s, t in zip(source, target):
                idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(self.noise_rate * len(idx))
                noisy_idx = np.random.choice(idx, n_noisy, replace=False)
                for i in noisy_idx:
                    self.targets[i] = t
        elif self.noise_rate > 0:
            for i in range(len(self.targets)):
                if np.random.rand() < self.noise_rate:
                    orig = self.targets[i]
                    choices = list(range(10))
                    choices.remove(orig)
                    self.targets[i] = np.random.choice(choices)


def get_noisy_cifar10_loaders(root='./data', noise_rate=0.0, asym=False,
                               batch_size=64, val_size=1000,
                               use_random_crop=False, use_random_flip=False, use_normalization=False,
                               num_workers=4, seed=42):

    transform_list = []
    if use_random_crop:
        transform_list.append(transforms.RandomCrop(32, padding=4))
    if use_random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    if use_normalization:
        transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                   (0.2023, 0.1994, 0.2010)))
    train_transform = transforms.Compose(transform_list)

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]) if use_normalization else transforms.ToTensor()

    if noise_rate > 0:
        full_train = CIFAR10Noisy(root=root, train=True, download=True,
                                   transform=train_transform,
                                   noise_rate=noise_rate, asym=asym, seed=seed)
    else:
        full_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                                  transform=train_transform)

    total = len(full_train)
    val_size = min(val_size, total)
    train_size = total - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True,
                                            transform=base_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

class CIFAR10Model(nn.Module):
    def __init__(self, use_dropout=True, dropout_rate=0.2, use_batchnorm=False, use_stride_downsampling=False, use_gap=False):
        super().__init__()
        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.use_stride_downsampling = use_stride_downsampling

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2 if use_stride_downsampling else 1)
        self.bn1_2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool1 = nn.Identity() if use_stride_downsampling else nn.MaxPool2d(2, 2)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2 if use_stride_downsampling else 1)
        self.bn2_2 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.pool2 = nn.Identity() if use_stride_downsampling else nn.MaxPool2d(2, 2)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2 if use_stride_downsampling else 1)
        self.bn3_2 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.pool3 = nn.Identity() if use_stride_downsampling else nn.MaxPool2d(2, 2)

        self.use_gap = use_gap
        if self.use_gap:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(256, 10)
        else:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, 10)

        layers_to_init = [
            self.conv1_1, self.conv1_2,
            self.conv2_1, self.conv2_2,
            self.conv3_1, self.conv3_2,
        ]

        if self.use_gap:
            layers_to_init.append(self.classifier)
        else:
            layers_to_init.extend([self.fc1, self.fc2])

        for layer in layers_to_init:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu' if layer != self.fc2 else 'linear')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout(x)

        if self.use_gap:
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        return x

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        log_probs = nn.functional.log_softmax(x, dim=-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (x.size(1) - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, A=-4, alpha=0.1, beta=1.0):
        super().__init__()
        self.A = A
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, target):
        ce = nn.functional.cross_entropy(x, target, reduction="none")

        pred_softmax = nn.functional.softmax(x, dim=1)
        target_onehot = nn.functional.one_hot(target, num_classes=x.size(1)).float()
        target_log = torch.log(target_onehot)
        target_log = torch.nan_to_num(target_log, nan=self.A, neginf=self.A)

        rce = -torch.sum(pred_softmax * target_log, dim=1)
        loss = self.alpha * ce + self.beta * rce

        return torch.mean(loss)


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
    use_random_crop = False #this is the translation agumentation that they are talking about
    use_random_flip = False
    use_normalization = False
    use_label_smoothing = False
    use_SymmetricCrossEntropy = True
    use_gap = False

    use_dropout = True
    dropout_rate = 0.3

    use_batchnorm = False
    use_stride_downsampling = False  # set False to use traditional max pooling

    use_scheduler = False #this gives the cosine with warmup
    use_warm_restarts = False  #this is active if we want the cosine with restarts
    warmup_epochs = 5
    total_epochs = 100
    cosine_anneal_epochs = total_epochs - warmup_epochs

    noisy_labels = False
    noise_rate = 0.5
    asym = False

    if noisy_labels:
        train_loader, val_loader, test_loader = get_noisy_cifar10_loaders(
            root='./data', noise_rate=noise_rate, asym=asym,
            batch_size=64, val_size=1000,
            use_random_crop=use_random_crop, use_random_flip=use_random_flip,
            use_normalization=use_normalization, num_workers=4, seed=42
        )
    else:
        train_loader, val_loader, test_loader = load_data(
            batch_size=64,
            val_size=1000,
            use_random_crop=use_random_crop,
            use_random_flip=use_random_flip,
            use_normalization=use_normalization
        )

    model = CIFAR10Model(
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
        use_stride_downsampling=use_stride_downsampling,
        use_gap=use_gap
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=0,
        amsgrad=False
    )

    if use_scheduler:
        if use_warm_restarts:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-5
            )
        else:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_anneal_epochs, eta_min=1e-4)
            scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = None

    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    elif use_SymmetricCrossEntropy:
        criterion = SymmetricCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()

    train_acc, val_acc = [], []
    train_loss, val_loss = [], []

    for epoch in range(1, total_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        val_loss.append(va_loss)
        val_acc.append(va_acc)

        print(f'Epoch {epoch}/{total_epochs} — '
              f'Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} — '
              f'Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}')

    if scheduler:
        if use_warm_restarts:
            scheduler.step(epoch - 1)
        else:
            scheduler.step()


    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'> Final test accuracy: {test_acc * 100:.2f}%')

    plot_diagnostics(train_acc, val_acc, train_loss, val_loss)


if __name__ == '__main__':
    main()
