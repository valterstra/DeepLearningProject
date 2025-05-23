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
import torch.nn.functional as F



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


class VGG1Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



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
    def __init__(self, A=-6, alpha=0.1, beta=1.0):
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


def evaluate_val(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total



def evaluate_test_per_class(model, dataloader, criterion, device, num_classes=10):
    model.eval()
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            for i in range(len(y)):
                label = y[i].item()
                pred = preds[i].argmax().item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    class_accuracy = [c / t if t > 0 else 0.0 for c, t in zip(class_correct, class_total)]
    return class_accuracy



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


def plot_per_class_accuracy(class_accuracies_dict, num_classes=10):
    class_indices = list(range(num_classes))
    plt.figure(figsize=(7, 5))

    for label, accuracies in class_accuracies_dict.items():
        plt.plot(class_indices, accuracies, marker='o', label=label)

    plt.xlabel('Class')
    plt.ylabel('Test accuracy')
    plt.title('Per-class test accuracy across epochs')
    plt.ylim(0.0, 1.05)
    plt.xticks(class_indices)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_classwise_accuracy_over_epochs(per_class_accuracy_log):
    epochs = sorted(per_class_accuracy_log.keys())
    num_classes = len(per_class_accuracy_log[epochs[0]])
    acc_matrix = np.array([per_class_accuracy_log[ep] for ep in epochs])

    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        plt.plot(epochs, acc_matrix[:, c], linestyle='--', label=f"Class {c}", alpha=0.6)

    mean_acc = acc_matrix.mean(axis=1)
    std_acc = acc_matrix.std(axis=1)
    plt.plot(epochs, mean_acc, color='brown', linewidth=2.5, label='Overall')
    plt.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color='purple', alpha=0.2)

    plt.xlabel('Number of epochs')
    plt.ylabel('Class-wise test accuracy')
    plt.title('Class-wise accuracy vs. Epochs')
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right', ncol=2)
    plt.tight_layout()
    plt.show()


def run_experiment_for_alphas():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A_values = [-2, -4, -6, -8]
    interval = 5

    use_random_crop = True
    use_random_flip = True
    use_normalization = True
    use_label_smoothing = False
    use_SymmetricCrossEntropy = True
    sce_alpha = 1.0
    sce_beta = 1.0
    use_gap = False

    use_dropout = True
    dropout_rate = 0.3
    use_batchnorm = True
    use_stride_downsampling = False

    total_epochs = 100
    noisy_labels = True
    noise_rate = 0.4
    asym = False

    train_loader, val_loader, test_loader = get_noisy_cifar10_loaders(
        root='./data', noise_rate=noise_rate, asym=asym,
        batch_size=64, val_size=1000,
        use_random_crop=use_random_crop, use_random_flip=use_random_flip,
        use_normalization=use_normalization, num_workers=4, seed=42
    )

    accuracy_logs = {}

    for A in A_values:
        model = CIFAR10Model(
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            use_stride_downsampling=use_stride_downsampling,
            use_gap=use_gap
        ).to(device)

        criterion = SymmetricCrossEntropy(alpha=sce_alpha, beta=sce_beta, A=A)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=0,
            amsgrad=False
        )

        epoch_acc_log = {}

        for epoch in range(1, total_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            if epoch % interval == 0:
                test_loss, test_acc = evaluate_val(model, test_loader, criterion, device)
                epoch_acc_log[epoch] = test_acc
        accuracy_logs[A] = epoch_acc_log

    return accuracy_logs


def run_ablation_study_sce_terms():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ablation_settings = {
        "SCE (α=0.1, β=1.0)": (0.1, 1.0),
        "No RCE (β=0)": (0.1, 0.0),
        "No CE (α=0)": (0.0, 1.0),
        "Upscaled CE (α=2.0)": (2.0, 1.0),
        "Upscaled RCE (β=5.0)": (0.1, 5.0)
    }

    total_epochs = 100
    interval = 5
    noise_rate = 0.4
    asym = False
    batch_size = 64
    val_size = 1000
    seed = 42

    accuracy_logs = {}

    for label, (alpha, beta) in ablation_settings.items():
        print(f"\nRunning: {label}")
        train_loader, val_loader, test_loader = get_noisy_cifar10_loaders(
            root='./data', noise_rate=noise_rate, asym=asym,
            batch_size=batch_size, val_size=val_size,
            use_random_crop=True, use_random_flip=True,
            use_normalization=True, num_workers=4, seed=seed
        )

        model = CIFAR10Model(
            use_dropout=True,
            dropout_rate=0.3,
            use_batchnorm=True,
            use_stride_downsampling=False,
            use_gap=False
        ).to(device)

        criterion = SymmetricCrossEntropy(alpha=alpha, beta=beta, A=-6)

        optimizer = optim.AdamW(
          model.parameters(),
          lr=0.001,
          betas=(0.9, 0.999),
          eps=1e-7,
          weight_decay=0,
          amsgrad=False
        )

        epoch_acc_log = {}

        for epoch in range(1, total_epochs + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            if epoch % interval == 0:
                _, test_acc = evaluate_val(model, test_loader, criterion, device)
                epoch_acc_log[epoch] = test_acc
                print(f"  Epoch {epoch}: Test Acc = {test_acc:.4f}")

        accuracy_logs[label] = epoch_acc_log

    plot_ablation_sce_terms(accuracy_logs)


def plot_ablation_sce_terms(accuracy_logs):
    plt.figure(figsize=(9, 6))

    for label, acc_log in accuracy_logs.items():
        epochs = list(acc_log.keys())
        accs = list(acc_log.values())
        plt.plot(epochs, accs, marker='o', label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Ablation Study of SCE Components")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(title="Loss Variant", fontsize=10)
    plt.tight_layout()
    plt.show()



def plot_sce_alpha_comparison(accuracy_logs):
    plt.figure(figsize=(8, 6))
    for alpha, acc_log in accuracy_logs.items():
        epochs = list(acc_log.keys())
        accs = list(acc_log.values())
        plt.plot(epochs, accs, marker='o', label=f"A={alpha}")

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("SCE: Test Accuracy vs Epochs for Different A Values")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()



def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_baseline_cnn = False

    # augmentations
    use_random_crop = True #this is the translation agumentation that they are talking about
    use_random_flip = True
    use_normalization = True

    use_label_smoothing = False
    use_SymmetricCrossEntropy = False
    sce_alpha = 0.1
    sce_beta = 1.0
    sce_A = -4
    use_gap = False

    use_dropout = True
    dropout_rate = 0.3

    use_batchnorm = True
    use_stride_downsampling = False  # set False to use traditional max pooling

    use_scheduler = True #this gives the cosine with warmup
    use_warm_restarts = False  #this is active if we want the cosine with restarts
    warmup_epochs = 5
    total_epochs = 100
    cosine_anneal_epochs = total_epochs - warmup_epochs

    noisy_labels = False
    noise_rate = 0.40
    asym = False

    #accuracy_logs = run_experiment_for_alphas()
    #plot_sce_alpha_comparison(accuracy_logs)
    #run_ablation_study_sce_terms()


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

    if use_baseline_cnn:
        model = VGG1Block().to(device)
    else:
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
        weight_decay=0.01,
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
        criterion = SymmetricCrossEntropy(alpha=sce_alpha, beta=sce_beta, A=sce_A)
    else:
        criterion = nn.CrossEntropyLoss()


    train_acc, val_acc = [], []
    train_loss, val_loss = [], []

    epoch_checkpoints = [10, 50, 100]
    class_acc_over_epochs = {}
    classwise_accuracy_log = {}

    plot_class_wise_acc = False
    interval_of_plot = 5


    for epoch in range(1, total_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate_val(model, val_loader, criterion, device)

        if plot_class_wise_acc and epoch % interval_of_plot == 0:
            test_class_acc = evaluate_test_per_class(model, test_loader, criterion, device)
            classwise_accuracy_log[epoch] = test_class_acc

            if epoch in epoch_checkpoints:
                test_class_acc = evaluate_test_per_class(model, test_loader, criterion, device)
                class_acc_over_epochs[f"Epoch {epoch}"] = test_class_acc

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

    if plot_class_wise_acc:
      plot_per_class_accuracy(class_acc_over_epochs)

    if plot_class_wise_acc:
        plot_classwise_accuracy_over_epochs(classwise_accuracy_log)

    test_loss, test_acc = evaluate_val(model, test_loader, criterion, device)
    print(f'> Final test accuracy: {test_acc * 100:.2f}%')

    plot_diagnostics(train_acc, val_acc, train_loss, val_loss)


if __name__ == '__main__':
    main()
