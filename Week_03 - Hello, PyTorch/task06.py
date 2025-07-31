import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize


def training(model, dataloader, criterion, optimizer, message, epochs=20):
    print(f'Training model {message}')

    losses = []

    for epoch in range(20):
        curr_loss = 0.0
        for batch_inputs, batch_labels in tqdm(dataloader):
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
        avg_loss = curr_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')
    return losses


def main():

    data = pd.read_csv('../DATA/ds_salaries.csv')

    features_list = data[[
        'experience_level', 'employment_type', 'remote_ratio', 'company_size'
    ]]
    targets_list = data[['salary_in_usd']]

    non_numerical_features = features_list[[
        'experience_level', 'employment_type', 'company_size'
    ]]
    numerical_features = features_list[['remote_ratio']]

    #I'm starting the encoding here and above I extracted the non-numerical features
    encoder = OneHotEncoder(sparse_output=False)
    non_numerical_features_encoded = encoder.fit_transform(
        non_numerical_features)

    result = np.hstack(
        (non_numerical_features_encoded, numerical_features.values))

    #Normalization
    normalized_features = normalize(result, axis=0)
    normalized_targets = normalize(targets_list, axis=0)

    features_tensor = torch.tensor(normalized_features, dtype=torch.float32)
    targets_tensor = torch.tensor(normalized_targets, dtype=torch.float32)

    dataset = TensorDataset(features_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    lr = 0.001
    criterion = nn.MSELoss()

    #Defining here 2 layers
    model_sigmoid = nn.Sequential(
        nn.Linear(in_features=features_tensor.shape[1], out_features=24),
        nn.Sigmoid(), nn.Linear(in_features=24, out_features=1))
    optimizer_sigmoid = optim.AdamW(model_sigmoid.parameters(), lr)
    losses_sigmoid = training(model_sigmoid,
                              dataloader,
                              criterion,
                              optimizer_sigmoid,
                              'nn_with_sigmoid',
                              epochs=20)

    model_relu = nn.Sequential(
        nn.Linear(in_features=features_tensor.shape[1], out_features=24),
        nn.ReLU(), nn.Linear(in_features=24, out_features=1))
    optimizer_relu = optim.AdamW(model_relu.parameters(), lr)
    losses_relu = training(model_relu,
                           dataloader,
                           criterion,
                           optimizer_relu,
                           'nn_with_relu',
                           epochs=20)

    model_lrelu = nn.Sequential(
        nn.Linear(in_features=features_tensor.shape[1], out_features=24),
        nn.LeakyReLU(negative_slope=0.05),
        nn.Linear(in_features=24, out_features=1))
    optimizer_lrelu = optim.AdamW(model_lrelu.parameters(), lr)
    losses_lrelu = training(model_lrelu,
                            dataloader,
                            criterion,
                            optimizer_lrelu,
                            'nn_with_leakyrelu',
                            epochs=20)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    axes[0].plot(losses_sigmoid)
    axes[0].set_title('nn_with_sigmoid')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Training Loss')
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    axes[1].plot(losses_relu)
    axes[1].set_title('nn_with_relu')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Training Loss')
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    axes[2].plot(losses_lrelu)
    axes[2].set_title('nn_with_leakyrelu')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Average Training Loss')
    axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.tight_layout()  # Make sure nothing overlaps
    plt.show()


if __name__ == "__main__":
    main()
