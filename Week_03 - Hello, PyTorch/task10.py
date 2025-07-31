import pandas as pd
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics
import numpy as np


def training(model, training_dataloader, validation_loader, test_loader,
             criterion, optimizer, epochs):

    train_losses = []
    validation_losses = []
    train_metrics = []
    validation_metrics = []

    train_f1_metric = torchmetrics.F1Score(task='binary')
    val_f1_metric = torchmetrics.F1Score(task='binary')

    for epoch in range(epochs):
        epoch_training_loss = 0.0
        epoch_validation_loss = 0.0

        train_f1_metric.reset()
        val_f1_metric.reset()

        model.train()
        curr_loss = 0.0
        for batch_inputs, batch_labels in tqdm(training_dataloader):
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            train_f1_metric.update(predictions, batch_labels.unsqueeze(1))
            loss = criterion(predictions, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
        epoch_training_loss = curr_loss / len(training_dataloader)
        train_losses.append(epoch_training_loss)
        train_metrics.append(train_f1_metric.compute().item())

        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_inputs, batch_labels in tqdm(validation_loader):
                predictions = model(batch_inputs)
                val_f1_metric.update(predictions, batch_labels.unsqueeze(1))
                loss = criterion(predictions, batch_labels.unsqueeze(1))
                validation_loss += loss.item()

        epoch_validation_loss = validation_loss / len(validation_loader)
        validation_losses.append(epoch_validation_loss)
        validation_metrics.append(val_f1_metric.compute().item())

        print(f'Epoch [{epoch+1}/30]:')
        print(f'  Average training Loss: {epoch_training_loss}')
        print(f'  Average validation Loss: {epoch_validation_loss}')
        print(f'  Training metric score: {train_f1_metric.compute().item()}')
        print(f'  Validation metric score: {val_f1_metric.compute().item()}')

    model.eval()
    test_loss = 0.0
    test_f1_metric = torchmetrics.F1Score(task='binary')

    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(test_loader):
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_labels.unsqueeze(1))
            test_loss += loss.item()
            test_f1_metric.update(predictions, batch_labels.unsqueeze(1))

    avg_test_loss = test_loss / len(test_loader)
    print(f"Average test loss: {avg_test_loss}")
    print(f"Test metric Score: {test_f1_metric.compute().item()}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(train_losses, label='Training Loss')
    axes[0].plot(validation_losses, label='Validation loss per epoch')
    axes[0].set_title('Loss per epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(train_metrics, label='Training')
    axes[1].plot(validation_metrics, label='Validation')
    axes[1].set_title('Metric per epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric')
    axes[1].legend()

    plt.show()


def print_information(data, name):

    info_counts = data['Potability'].value_counts().to_frame('counts')
    info_counts['proportions'] = (info_counts / len(data)).round(6)

    print(f"Distribution of target values in {name} set:")
    print(info_counts)


def main():

    load_train_data = pd.read_csv('../DATA/water_train.csv')
    load_test_data = pd.read_csv('../DATA/water_test.csv')

    size = len(load_test_data)
    validation_size = size // 2
    test_size = size - validation_size

    data_validation, data_test = random_split(load_test_data,
                                              [validation_size, test_size])

    #Vrushtame Subset, sledovatelno pak trqbva da go convert-nem kum DataFrame, za da moga da izprintq distribution-a
    data_validation_df = load_test_data.iloc[data_validation.indices]
    data_test_df = load_test_data.iloc[data_test.indices]


    print_information(load_train_data, "train")
    print_information(data_validation_df, "validation")
    print_information(data_test_df, "test")
    
    data_train_features = load_train_data.drop(columns=['Potability'])
    data_train_target = load_train_data['Potability']

    data_validation_features = data_validation_df.drop(columns=['Potability'])
    data_validation_target = data_validation_df['Potability']

    data_test_features = data_test_df.drop(columns=['Potability'])
    data_test_target = data_test_df['Potability']

    data_train_dataset = TensorDataset(
        torch.tensor(data_train_features.values, dtype=torch.float32),
        torch.tensor(data_train_target.values, dtype=torch.float32))
    data_train_dataloader = DataLoader(data_train_dataset,
                                       batch_size=8,
                                       shuffle=True)

    data_test_dataset = TensorDataset(
        torch.tensor(data_test_features.values, dtype=torch.float32),
        torch.tensor(data_test_target.values, dtype=torch.float32))
    data_test_dataloader = DataLoader(data_test_dataset,
                                      batch_size=8,
                                      shuffle=False)

    data_validation_dataset = TensorDataset(
        torch.tensor(data_validation_features.values, dtype=torch.float32),
        torch.tensor(data_validation_target.values, dtype=torch.float32))
    data_validation_dataloader = DataLoader(data_validation_dataset,
                                            batch_size=8,
                                            shuffle=False)

    model = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 16),
                          nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    criterion = nn.BCELoss()

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr)

    training(model, data_train_dataloader, data_validation_dataloader,
             data_test_dataloader, criterion, optimizer, 30)


if __name__ == "__main__":
    main()
