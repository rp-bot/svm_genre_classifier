from torch.utils.data import Dataset, Sampler, DataLoader, random_split
import pandas as pd
import torchaudio
from torchaudio.transforms import MFCC, MelSpectrogram, Resample
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from _dataloader import clean_and_split_data
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torchsummary import summary


class MLP(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 512)
        # self.fc2 = nn.Linear(512, 256)

        self.fc4 = nn.Linear(512, num_classes)

        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.flatten(x)
        x = self.relu(self.batchnorm1(self.fc1(x)))
        # x = self.relu(self.batchnorm2(self.fc2(x)))
        # x = self.relu(self.batchnorm3(self.fc3(x)))
        x = self.fc4(x)
        return self.softmax(x)


class PatchBanksDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        transformation,
        resampler,
        target_sample_rate,
        duration,
        device,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.input_duration = duration
        self.device = device
        self.tranformation = transformation.to(self.device)
        self.resampler = resampler.to(self.device)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self.annotations.iloc[index, 1]
        label = self.annotations.iloc[index, 3]
        signal, sr = torchaudio.load(audio_sample_path, normalize=True)
        signal = signal.to(self.device)

        # trim to only 5 seconds
        num_samples = self.input_duration * self.target_sample_rate
        signal = signal[:, :num_samples]

        fade_samples = int(0.25 * self.target_sample_rate)
        fade_curve = torch.linspace(1, 0, fade_samples).to(signal.device)

        signal[:, -fade_samples:] *= fade_curve

        # Ensure signal is stereo
        signal = self.resampler(signal)
        signal = self.tranformation(signal)

        if signal.shape[0] != 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        signal = signal.transpose(2, 1)

        # signal = torch.log1p(signal)
        return signal, label


def make_torch_dataset(experiment_num):
    X_train, X_test, y_train, y_test, label_encoder = clean_and_split_data()
    # plot_feature_vector_distribution(X_train_scaled, y_train)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    device = torch.device("cuda")

    if experiment_num == 1:

        X_train = X_train[:, :7]
        X_test = X_test[:, :7]
        print("Experiment 1 Train Set: ", X_train.shape)
        print("Experiment 1 Test Set: ", X_test.shape)

    elif experiment_num == 2:
        X_train = X_train[:, 8:]
        X_test = X_test[:, 8:]
        print("Experiment 2 Train Set: ", X_train.shape)
        print("Experiment 2 Test Set: ", X_test.shape)

    elif experiment_num == 3:
        X_train = X_train[:, :15]
        X_test = X_test[:, :15]
        print("Experiment 3 Train Set: ", X_train.shape)
        print("Experiment 3 Test Set: ", X_test.shape)

    elif experiment_num == 4:
        X_train = X_train[:, 15:]
        X_test = X_test[:, 15:]
        print("Experiment 4 Train Set: ", X_train.shape)
        print("Experiment 4 Test Set: ", X_test.shape)

    elif experiment_num == 5:
        print("Experiment 5 Train Set: ", X_train.shape)
        print("Experiment 5 Test Set: ", X_test.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    return train_dataset, test_dataset


if __name__ == "__main__":
    device = torch.device("cuda")

    # train_dataset, test_dataset = make_torch_dataset()

    # # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # print(train_dataset[0][0].shape)

    model = MLP(in_features=7, num_classes=4)

    summary(model.cuda(), (7,))
