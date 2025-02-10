import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torchaudio.transforms import MFCC, Resample
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split

# from _dataset import PatchBanksDataset, StratifiedBatchSampler
from MLP import MLP, make_torch_dataset
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
from datetime import datetime


class_mappings = [
    "tr-909",
    "hip-hop",
    "pop rock",
    "samba",
]


def evaluate(model, test_data_loader, loss_fn, device, num_classes):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_data_loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

            predicted = torch.argmax(predictions, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            all_targets.append(targets)
            all_preds.append(predicted)

    avg_loss = total_loss / len(test_data_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    all_targets = torch.cat(all_targets).cpu()
    all_preds = torch.cat(all_preds).cpu()

    cm_metric = ConfusionMatrix(
        num_classes=num_classes, normalize="none", task="multiclass"
    )
    cm = cm_metric(all_preds, all_targets)

    cm_np = cm.numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_np,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_mappings,
        yticklabels=class_mappings,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    current_time = datetime.now()
    time_str = current_time.strftime("%m_%d_%H_%M")
    plt.savefig(f"plots/metrics/confusion_matrix_{time_str}.png")


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    running_loss = 0.0
    correct = 0
    total = 1e-10
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.argmax(predictions, dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def train(
    model,
    loss_fn,
    optimizer,
    scheduler,
    device,
    epochs,
    train_labels=None,
    train_dataset=None,
    batch_size=None,
    train_dataloader=None,
):

    model.train()
    loss_during_training = []
    accuracy_during_training = []

    for i in range(epochs):
        # train_sampler = StratifiedBatchSampler(train_labels, batch_size, seed=42)
        # train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
        print(f"Epoch {i+1}")
        avg_loss, accuracy = train_one_epoch(
            model, train_dataloader, loss_fn, optimizer, device
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr}")
        print("----------------------------")
        loss_during_training.append(avg_loss)
        accuracy_during_training.append(accuracy)
    print("Training is Done")

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 6))
    # loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss_during_training, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss During Training")
    plt.legend()
    # accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracy_during_training, label="Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy During Training")
    plt.legend()
    current_time = datetime.now()
    time_str = current_time.strftime("%m_%d_%H_%M")
    plt.savefig(f"plots/metrics/training_metrics_{time_str}.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    annotations = "data/annotations.csv"
    audio_dir = "data"
    SAMPLE_RATE = 44_100
    TARGET_SAMPLE_RATE = 22_050
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 1e-3
    SEED = 42

    # mfcc_transform = MFCC(
    #     sample_rate=SAMPLE_RATE,
    #     n_mfcc=20,
    #     melkwargs={
    #         "n_fft": 1024,
    #         "hop_length": 512,
    #         "n_mels": 40,
    #     },
    # )

    # resampler = Resample(SAMPLE_RATE, TARGET_SAMPLE_RATE)

    train_dataset, test_dataset = make_torch_dataset()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # train_size = int(0.7 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # train_labels = [dataset.annotations.iloc[i, 3] for i in train_dataset.indices]
    # test_labels = [dataset.annotations.iloc[i, 3] for i in test_dataset.indices]

    # train_sampler = StratifiedBatchSampler(train_labels, BATCH_SIZE, seed=42)
    # test_sampler = StratifiedBatchSampler(test_labels, BATCH_SIZE, seed=42)

    # train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    mlp_network = MLP(in_features=49, num_classes=4).to(device)
    # mlp_network.load_state_dict(torch.load("trained_models/MLP_02_09_18_09.pth"))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_network.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    print("Training Model")
    train(
        mlp_network,
        loss_fn,
        optimizer,
        scheduler,
        device,
        epochs=EPOCHS,
        # train_labels=train_labels,
        # train_dataset=train_dataset,
        # batch_size=BATCH_SIZE,
        train_dataloader=train_loader,
    )
    print("============================")

    print("Final Evaluation on Test Set:")
    evaluate(mlp_network, test_loader, loss_fn, device, num_classes=4)
    print("============================")

    print("saving Model")
    current_time = datetime.now()
    time_str = current_time.strftime("%m_%d_%H_%M")
    torch.save(mlp_network.state_dict(), f"trained_models/MLP_{time_str}.pth")
    print("model saved")
