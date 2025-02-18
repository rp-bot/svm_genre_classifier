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
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

SAMPLE_RATE = 44_100
TARGET_SAMPLE_RATE = 22_050
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
SEED = 42


class_mappings = [
    "tr-909",
    "hip-hop",
    "pop rock",
    "samba",
]


def plot_metrics(test_metrics, metric_names, model_name, plot_name, folder):
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars2 = ax.bar(x + width / 2, test_metrics, width, label="Test", color="orange")

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.savefig(f"plots/metrics/MLP/{folder}/{plot_name}.png")


def evaluate(
    model,
    test_data_loader,
    loss_fn,
    device,
    num_classes,
    experiment_num,
):
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
    print(f"Validation Loss: {avg_loss:.4f},")

    all_targets = torch.cat(all_targets).cpu()
    all_preds = torch.cat(all_preds).cpu()

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    recall = recall_score(all_targets, all_preds, average="weighted")
    precision = precision_score(all_targets, all_preds, average="weighted")

    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"Precision: {precision:.4f}")

    test_metrics = [accuracy, f1, recall, precision]
    print(test_metrics)
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    plot_metrics(
        test_metrics=test_metrics,
        metric_names=metric_names,
        model_name=f"MLP Experiment {experiment_num}",
        plot_name=f"experiment_{experiment_num}_MLP_metrics",
        folder=f"experiment_{experiment_num}",
    )

    cm_metric = ConfusionMatrix(
        num_classes=num_classes, normalize="true", task="multiclass"
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
    plt.title(f"Experiment {experiment_num} Normalized Confusion Matrix")
    plt.tight_layout()
    current_time = datetime.now()
    plt.savefig(
        f"plots/metrics/MLP/experiment_{experiment_num}/experiment_{experiment_num}confusion_matrix.png"
    )


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
    experiment_num,
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
    plt.savefig(f"plots/metrics/MLP/experiment_{experiment_num}/training_metrics.png")


def experiments(experiment_number):
    train_dataset, test_dataset = make_torch_dataset(experiment_number)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    num_in_features = train_dataset[0][0].shape[0]
    mlp_network = MLP(in_features=num_in_features, num_classes=4).to(device)
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
        experiment_num=experiment_number,
        epochs=EPOCHS,
        # train_labels=train_labels,
        # train_dataset=train_dataset,
        # batch_size=BATCH_SIZE,
        train_dataloader=train_loader,
    )
    print("============================")
    print("Final Evaluation on Test Set:")
    evaluate(
        mlp_network,
        test_loader,
        loss_fn,
        device,
        num_classes=4,
        experiment_num=experiment_number,
    )
    print("============================")
    current_time = datetime.now()
    time_str = current_time.strftime("%m_%d_%H_%M")


if __name__ == "__main__":
    device = torch.device("cuda")

    torch.cuda.empty_cache()

    annotations = "data/annotations.csv"
    audio_dir = "data"

    # mfcc_transform = MFCC(
    #     sample_rate=SAMPLE_RATE,
    #     n_mfcc=20,
    #     melkwargs={
    #         "n_fft": 1024,
    #         "hop_length": 512,
    #         "n_mels": 40,
    #     },
    # )
    experiments(5)
    # resampler = Resample(SAMPLE_RATE, TARGET_SAMPLE_RATE)

    # train_dataset, test_dataset = make_torch_dataset(1)

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # print(train_dataset[0][0].shape[0])

    # train_size = int(0.7 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # train_labels = [dataset.annotations.iloc[i, 3] for i in train_dataset.indices]
    # test_labels = [dataset.annotations.iloc[i, 3] for i in test_dataset.indices]

    # train_sampler = StratifiedBatchSampler(train_labels, BATCH_SIZE, seed=42)
    # test_sampler = StratifiedBatchSampler(test_labels, BATCH_SIZE, seed=42)

    # train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
