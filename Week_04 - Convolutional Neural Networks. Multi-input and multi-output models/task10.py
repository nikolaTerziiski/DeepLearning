import os
import glob
import re
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Precision, Recall, F1Score
import torch.optim as optim
from PIL import Image




class OmniDataset(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        self.samples = []
        self.alphabets = sorted(os.listdir(path))
        self.alphabet_to_idx = {name: idx for idx, name in enumerate(self.alphabets)}
        self.num_alphabets = len(self.alphabets)

        for alphabet in self.alphabets:
            alphabet_path = os.path.join(path, alphabet)
            for character in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character)
                
                if not os.path.isdir(character_path):
                    continue
                
                for img_path in os.scandir(character_path):
                    #Pravq proverkata, zashtoto mi davashe greshka za DS_Store nqkakuv fail, che e tip hidden
                    if  img_path.is_file() and img_path.name.endswith('.png'):
                        image_filename = os.path.basename(img_path)
                        match = re.match(r"(\d+)_\d+\.png", image_filename)
                        character_label = int(match.group(1)) if match else -1
                        self.samples.append((img_path, alphabet, character_label))
                    
        #I tried to other way with ImageFolder, but not sure if this is actually wanted here, spent a lot of time researching that algorithm for the folders
        #Please, if the other way is wanted, tell me and I will send it

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, alphabet_name, character_label = self.samples[idx]

        one_hot = torch.zeros(self.num_alphabets, dtype=torch.float32)
        
        img = Image.open(image_path)
        img = self.transform(img)
        
        alpha_idx = self.alphabet_to_idx[alphabet_name]
        one_hot[alpha_idx] = 1.0

        return img, one_hot, character_label - 1

class Net(nn.Module):
    def __init__(self, alphabets_count=30, characters=964):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        self.alphabet = nn.Linear(alphabets_count, 64)
        
        
        self.fc1 = nn.Linear(64 * 16 * 16 + 64, 256)
        self.fc2 = nn.Linear(256, characters)
        
    def forward(self, x_img, x_alphabet):
        x = F.relu(self.conv1(x_img))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x,1)
        
        alpha = F.relu(self.alphabet(x_alphabet))
        
        x = torch.cat((x, alpha), dim=1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

def training(model, training_loader, validation_loader, optimizer, criterion, num_epoch):
    
    f1_training_score = F1Score(task="multiclass", num_classes=964, average="macro")
    f1_validation_score = F1Score(task="multiclass", num_classes=964, average="macro")
    
    f1_training_scores = []
    f1_validation_scores = []
    
    training_model_loss = []
    validation_model_loss = []
    
    for epoch in range(num_epoch):
        model.train()
        training_loss = []
        f1_training_score.reset()
        
        for x_img, x_alpha, index in tqdm(training_loader):
            output = model(x_img, x_alpha)
            loss = criterion(output, index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss)
            
            preds = torch.argmax(output, dim=1)
            f1_training_score.update(preds, index)
        
        f1_validation_score.reset()
        model.eval()
        total_val_loss = []
        with torch.no_grad():
            for x_img, x_alpha, index in tqdm(validation_loader):
                output = model(x_img, x_alpha)
                loss = criterion(output, index)
                
                
                preds = torch.argmax(output, dim=1)
                total_val_loss.append(loss)
                f1_validation_score.update(preds, index)
                
        f1_training_scores.append(f1_training_score.compute())
        f1_validation_scores.append(f1_validation_score.compute())
        
        average_training_loss = sum(training_loss) / len(training_loss)
        average_validation_loss = sum(total_val_loss) / len(total_val_loss)
        
        training_model_loss.append(average_training_loss)
        validation_model_loss.append(average_validation_loss)
        
        print(f"Epoch [{epoch+1}/{num_epoch}]:")
        print(f" Average training loss: {average_training_loss}") 
        print(f" Average validation loss:{average_validation_loss}")
        print(f" Training metric score: {f1_training_score.compute()}")
        print(f" Validation metric score: {f1_validation_score.compute()}")
        
        
    
def main():
    
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64,64))
    ])
    
    train_dataset = OmniDataset("../DATA/omniglot_train", transformer)
    validation_dataset = OmniDataset("../DATA/omniglot_test", transformer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True)
    
    model = Net()
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), learning_rate)
    
    training(model, train_loader, validation_loader, optimizer, criterion, num_epoch=5)


if __name__ == "__main__":
    main()