import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


#TOOK THAT DEFINITION FROM https://medium.com/we-talk-data/how-to-set-random-seeds-in-pytorch-and-tensorflow-89c5f8e80ce4
# Define the seed value
seed = 42

# Set seed for PyTorch
torch.manual_seed(seed)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


def train_model(dataloader_train,
                optimizer,
                criterion,
                net,
                num_epochs,
                create_plot=False):

    train_losses = []

    net.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_training_loss = 0.0
        curr_loss = 0.0

        for batch_inputs, batch_labels in dataloader_train:
            optimizer.zero_grad()
            predictions = net(batch_inputs)
            loss = criterion(predictions, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()

        epoch_training_loss = curr_loss / len(dataloader_train)
        train_losses.append(epoch_training_loss)

    print(f"Average loss: {sum(train_losses) / len(train_losses)}")

    if create_plot:
        plt.plot(range(1, num_epochs + 1), train_losses, label='Loss')
        plt.title("Loss per epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


def main():

    train_data = pd.read_csv('../DATA/water_train.csv').to_numpy()
    test_data = pd.read_csv('../DATA/water_test.csv').to_numpy()

    train_features = train_data[:, :-1]
    train_target = train_data[:, -1]

    test_features = test_data[:, :-1]
    test_target = test_data[:, -1]

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_target, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_features, dtype=torch.float32),
        torch.tensor(test_target, dtype=torch.float32))

    dataloader_train = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=16, shuffle=True)

    lr = 0.001

    criterion = nn.BCELoss()

    optimizer = None
    for iter_optimizer in [
            torch.optim.SGD,
            torch.optim.RMSprop,
            torch.optim.Adam,
            torch.optim.AdamW,
    ]:
        net = Net()
        optimizer = iter_optimizer(net.parameters(), lr=lr)
        print(f"Using the {optimizer.__class__.__name__} optimizer:")
        train_model(dataloader_train, optimizer, criterion, net, num_epochs=10)

    net = Net()
    # I defined here new optimizer, because if i use the one from the for loop the .step method will update the old Net() from the loop
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    train_model(dataloader_train,
                optimizer,
                criterion,
                net,
                num_epochs=1000,
                create_plot=True)

    net.eval()
    f1 = torchmetrics.F1Score(task='binary', threshold=0.5)
    with torch.no_grad():
        for batch_features, batch_labels in dataloader_test:
            f1.update(net(batch_features), batch_labels.unsqueeze(1).long())
    print(f"F1 score on test set: {f1.compute().item()}")

    return


if __name__ == "__main__":
    main()
