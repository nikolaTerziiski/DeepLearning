from PIL import Image
from torchvision import transforms
from torch.utils import data
from torchvision.datasets import ImageFolder
from torch import nn
from tqdm import tqdm

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


def train_model(dataloader_train, optimizer, criterion, net, num_epochs):

    net.train()
    epoch_training_loss = []
    for epoch in range(num_epochs):
        curr_loss = 0.0

        dataloader_train_batch = tqdm(dataloader_train,
                                      desc=f"Epoch {epoch+1}: ",
                                      unit="batch")
        for batch_inputs, batch_labels in dataloader_train_batch:
            optimizer.zero_grad()
            predictions = net(batch_inputs)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
        avg_loss = curr_loss / len(dataloader_train)
        epoch_training_loss.append(avg_loss)
        print(f"Average loss per batch: {avg_loss}")

    print(f"Average loss per epoch: {sum(epoch_training_loss) / num_epochs}")

    plt.plot(range(1, num_epochs + 1),
             epoch_training_loss,
             label='Loss per epoch')
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)

        self.classifier = nn.Linear(64 * 16 * 16, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.elu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        return self.classifier(x)


def main():

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    dataset_train = ImageFolder('../DATA/clouds/clouds_train',
                                transform=train_transform)

    dataloader_train = data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=16,
    )

    rand_indx = np.random.randint(0, len(dataset_train))
    image, label = dataset_train[rand_indx]
    image = image.permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)

    train_model(dataloader_train, optimizer, criterion, net, num_epochs=20)


if __name__ == "__main__":
    main()
