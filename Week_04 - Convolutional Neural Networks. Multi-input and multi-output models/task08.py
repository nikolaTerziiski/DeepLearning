from PIL import Image
from torchvision import transforms
from torch.utils import data
from torchvision.datasets import ImageFolder
from torch import nn
from tqdm import tqdm
from pprint import pprint

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time
import torch

from torchmetrics import Precision, Recall, F1Score


def train_model(dataloader_train, dataloader_test, dataset_test, optimizer, criterion, net, num_epochs):

    
    t0 = time.time()
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
        
    num_classes = 7
    precision_metric = Precision('multiclass', num_classes=num_classes, average='macro')
    recall_metric = Recall('multiclass', num_classes=num_classes, average='macro')
    f1_score = F1Score('multiclass', num_classes=num_classes, average='macro')
    
    f1_per_class = F1Score('multiclass',num_classes=num_classes, average='none')
    
    precision_metric.reset()
    recall_metric.reset()
    f1_score.reset()
    f1_per_class.reset()
    
    net.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader_test:
            logits = net(batch_inputs)
            preds = logits.argmax(dim=1)
            
            precision_metric.update(preds, batch_labels)
            recall_metric.update(preds, batch_labels)
            f1_score.update(preds, batch_labels)
            f1_per_class.update(preds, batch_labels)
            
    
                    
    prec_result = precision_metric.compute()
    rec_result = recall_metric.compute()
    f1_score_result = f1_score.compute()
    f1_per_class_result = f1_per_class.compute()
    t1 = time.time()
    
    print("Summary statistics:")
    print(f"Average loss per epoch: {sum(epoch_training_loss) / num_epochs}")
    print(f"Precision: {prec_result:.4f}")
    print(f"Recall: {rec_result:.4f}")
    print(f"F1 Score: {f1_score_result:.4f}")
    
    classes = dataset_test.classes
    f1_score_per_class = [ x.item() for x in f1_per_class_result]
    
    final_per_score = dict(zip(classes, f1_score_per_class))
    
    print("\nPer class F1 score:")
    pprint(final_per_score)
    print(f"Average time taken: {t1 - t0:.2f} seconds")


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
    
    dataset_test = ImageFolder('../DATA/clouds/clouds_test', transform=train_transform)
    dataloader_test = data.DataLoader(dataset_test, shuffle=True, batch_size=16)

    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)

    train_model(dataloader_train, dataloader_test, dataset_test, optimizer, criterion, net, num_epochs=20)
    

if __name__ == "__main__":
    main()
