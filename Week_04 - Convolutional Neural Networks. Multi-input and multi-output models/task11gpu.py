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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OmniDataset(Dataset):

    def __init__(self, path, transform):
        self.transform = transform
        self.samples = []
        self.alphabets = sorted(os.listdir(path))
        self.alphabet_to_idx = {
            name: idx
            for idx, name in enumerate(self.alphabets)
        }
        self.num_alphabets = len(self.alphabets)

        for alphabet in self.alphabets:
            alphabet_path = os.path.join(path, alphabet)
            for character in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character)

                if not os.path.isdir(character_path):
                    continue

                for img_path in os.scandir(character_path):
                    if img_path.is_file() and img_path.name.endswith('.png'):
                        image_filename = os.path.basename(img_path)
                        match = re.match(r"(\d+)_\d+\.png", image_filename)
                        character_label = int(match.group(1)) if match else -1

                        self.samples.append(
                            (img_path, alphabet, character_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, alphabet_name, character_label = self.samples[idx]

        one_hot = torch.zeros(self.num_alphabets, dtype=torch.float32)

        img = Image.open(image_path)
        img = self.transform(img)

        alpha_idx = self.alphabet_to_idx[alphabet_name]

        return img, alpha_idx, character_label - 1


class Net(nn.Module):

    def __init__(self, alphabets=30, characters=964):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(64 * 16 * 16, 256)

        self.fc2_char = nn.Linear(256, characters)
        self.fc2_alpha = nn.Linear(256, alphabets)

    def forward(self, x_img):
        x = F.relu(self.conv1(x_img))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)

        initial = self.fc1(x)

        x_alph = self.fc2_alpha(initial)
        x_char = self.fc2_char(initial)

        return x_alph, x_char


def training(model, training_loader, validation_loader, optimizer, criterion,
             num_epoch):

    f1_training_score_character = F1Score(task="multiclass",
                                          num_classes=964,
                                          average="macro").to(device)
    f1_validation_score_character = F1Score(task="multiclass",
                                            num_classes=964,
                                            average="macro").to(device)

    f1_training_score_alphabet = F1Score(task="multiclass",
                                         num_classes=30,
                                         average="macro").to(device)
    f1_validation_score_alphabet = F1Score(task="multiclass",
                                           num_classes=30,
                                           average="macro").to(device)

    f1_training_scores_character = []
    f1_validation_scores_character = []

    f1_training_scores_alphabet = []
    f1_validation_scores_alphabet = []

    training_model_loss = []
    validation_model_loss = []

    for epoch in range(num_epoch):
        model.train()
        training_loss = []
        f1_training_score_character.reset()
        f1_training_score_alphabet.reset()

        for x_img, x_alpha, index in tqdm(training_loader):

            x_img = x_img.to(device)
            x_alpha = x_alpha.to(device)
            index = index.to(device)

            alphabet_logit, character_logit = model(x_img)
            loss_alphabet_logit = criterion(alphabet_logit, x_alpha)
            loss_character_logit = criterion(character_logit, index)

            loss = 0.33 * loss_alphabet_logit + 0.67 * loss_character_logit
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss)

            preds_char = torch.argmax(character_logit, dim=1)
            preds_alph = torch.argmax(alphabet_logit, dim=1)

            f1_training_score_character.update(preds_char, index)
            f1_training_score_alphabet.update(preds_alph, x_alpha)

        f1_validation_score_character.reset()
        f1_validation_score_alphabet.reset()
        model.eval()
        total_val_loss = []
        with torch.no_grad():
            for x_img, x_alpha, index in tqdm(validation_loader):
                x_img = x_img.to(device)
                x_alpha = x_alpha.to(device)
                index = index.to(device)

                alphabet_logit, character_logit = model(x_img)
                loss_alphabet_logit = criterion(alphabet_logit, x_alpha)
                loss_character_logit = criterion(character_logit, index)

                loss = 0.33 * loss_alphabet_logit + 0.67 * loss_character_logit

                total_val_loss.append(loss)

                preds_char = torch.argmax(character_logit, dim=1)
                preds_alph = torch.argmax(alphabet_logit, dim=1)

                f1_validation_score_character.update(preds_char, index)
                f1_validation_score_alphabet.update(preds_alph, x_alpha)

        ##Here I append to the arrays the computed value of the F1 Scores for the Characters
        f1_training_scores_character.append(
            f1_training_score_character.compute().cpu().item())
        f1_validation_scores_character.append(
            f1_validation_score_character.compute().cpu().item())

        ##Here I append to the arrays the computed value of the F1 Scores for the Alphabets
        f1_training_scores_alphabet.append(
            f1_training_score_alphabet.compute().cpu().item())
        f1_validation_scores_alphabet.append(
            f1_validation_score_alphabet.compute().cpu().item())

        average_training_loss = (sum(training_loss) /
                                 len(training_loss)).cpu().item()
        average_validation_loss = (sum(total_val_loss) /
                                   len(total_val_loss)).cpu().item()

        training_model_loss.append(average_training_loss)
        validation_model_loss.append(average_validation_loss)

        print(f"Epoch [{epoch+1}/{num_epoch}]:")
        print(f" Average training loss: {average_training_loss}")
        print(f" Average validation loss:{average_validation_loss}")
        print(
            f" Training metric score character: {f1_training_score_character.compute()}"
        )
        print(
            f" Validation metric score character: {f1_validation_score_character.compute()}"
        )
        print(
            f" Training metric score alphabet: {f1_training_score_alphabet.compute()}"
        )
        print(
            f" Validation metric score alphabet: {f1_validation_score_alphabet.compute()}"
        )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

    ax1.plot(training_model_loss, label="Training loss")
    ax1.plot(validation_model_loss, label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(f1_training_scores_character, label="Train F1 Characters")
    ax2.plot(f1_validation_scores_character, label="Validation F1 Score")
    ax2.set_title("F1 Score over Epochs Characters")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend()

    ax3.plot(f1_training_scores_alphabet, label="Train F1 Alphabets")
    ax3.plot(f1_validation_scores_alphabet,
             label="Validation F1 Score Alphabets")
    ax3.set_title("F1 Score over Epochs - Alphabets")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1 Score")
    ax3.legend()

    plt.show()


def main():

    transformer = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((64, 64))])

    train_dataset = OmniDataset("../DATA/omniglot_train", transformer)
    validation_dataset = OmniDataset("../DATA/omniglot_test", transformer)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=512,
                                   shuffle=True)

    model = Net().to(device)
    print(device)
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), learning_rate)

    training(model,
             train_loader,
             validation_loader,
             optimizer,
             criterion,
             num_epoch=6)


if __name__ == "__main__":
    main()
