import os
import re
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics import F1Score
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
        one_hot[alpha_idx] = 1.0

        return img, one_hot, character_label - 1


class Net(nn.Module):

    def __init__(self, alphabets_count=30, characters=964):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(start_dim=1)

        self.alphabet = nn.Linear(alphabets_count, 64)

        self.fc1 = nn.Linear(64 * 16 * 16 + 64, 256)
        self.fc2 = nn.Linear(256, characters)

    def forward(self, x_img, x_alphabet):
        x = F.relu(self.conv1(x_img))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)

        alpha = F.relu(self.alphabet(x_alphabet))

        x = torch.cat((x, alpha), dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


def training(model, training_loader, validation_loader, optimizer, criterion,
             num_epoch):

    f1_training_score = F1Score(task="multiclass",
                                num_classes=964,
                                average="macro").to(device)
    f1_validation_score = F1Score(task="multiclass",
                                  num_classes=964,
                                  average="macro").to(device)

    f1_training_scores = []
    f1_validation_scores = []

    training_model_loss = []
    validation_model_loss = []

    for epoch in range(num_epoch):
        model.train()
        training_loss = []
        f1_training_score.reset()

        for x_img, x_alpha, index in tqdm(training_loader):

            x_img = x_img.to(device)
            x_alpha = x_alpha.to(device)
            index = index.to(device)

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
                x_img = x_img.to(device)
                x_alpha = x_alpha.to(device)
                index = index.to(device)

                output = model(x_img, x_alpha)
                loss = criterion(output, index)

                total_val_loss.append(loss)

                preds = torch.argmax(output, dim=1)
                f1_validation_score.update(preds, index)

        f1_training_scores.append(f1_training_score.compute().cpu().item())
        f1_validation_scores.append(f1_validation_score.compute().cpu().item())

        average_training_loss = (sum(training_loss) /
                                 len(training_loss)).cpu().item()
        average_validation_loss = (sum(total_val_loss) /
                                   len(total_val_loss)).cpu().item()

        training_model_loss.append(average_training_loss)
        validation_model_loss.append(average_validation_loss)

        print(f"Epoch [{epoch+1}/{num_epoch}]:")
        print(f" Average training loss: {average_training_loss}")
        print(f" Average validation loss:{average_validation_loss}")
        print(f" Training metric score: {f1_training_score.compute()}")
        print(f" Validation metric score: {f1_validation_score.compute()}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(training_model_loss, label="Training loss")
    ax1.plot(validation_model_loss, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(f1_training_scores, label="Train F1 Score")
    ax2.plot(f1_validation_scores, label="Validation F1 Score")
    ax2.set_title("F1 Score over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend()

    plt.show()


from collections import Counter
from torch.utils.data import DataLoader

# ... (OmniDataset и останалия ви код) ...

def print_class_distribution_via_dataloader(dataset, batch_size==512):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    counter = Counter()

    for imgs, one_hot_alpha, labels in loader:
        # labels е тензор с размер [batch_size]; правим .tolist() за удобство
        for lbl in labels.tolist():
            counter[lbl] += 1

    # Отпечатваме резултата в нарастващ ред на label-а
    for label, count in sorted(counter.items()):
        print(f"Label {label}: {count} samples")


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    train_dataset = OmniDataset("../DATA/omniglot_train", transformer)
    print("Class distribution in TRAIN set:")
    print_class_distribution_via_dataloader(train_dataset, batch_size=512)

    val_dataset = OmniDataset("../DATA/omniglot_test", transformer)
    print("\nClass distribution in VALIDATION set:")
    print_class_distribution_via_dataloader(val_dataset, batch_size=512)


# def main():

#     transformer = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Resize((64, 64))])

#     train_dataset = OmniDataset("../DATA/omniglot_train", transformer)
#     validation_dataset = OmniDataset("../DATA/omniglot_test", transformer)

#     train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
#     validation_loader = DataLoader(validation_dataset,
#                                    batch_size=512,
#                                    shuffle=True)
    
#     print(train_loader.__class__())
#     for i in range(100):
#         img, one_hot_alpha, label0 = train_dataset[i]
#         print(f"{i} - {one_hot_alpha.argmax().item()} - {label0}")
              
#     # print("Unique labels in dataset:",
#     #       sorted(set([label for _, _, label in train_dataset.samples])))

#     # model = Net().to(device)
#     # print(device)
#     # learning_rate = 0.001
#     # criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.AdamW(model.parameters(), learning_rate)

#     # training(model,
#     #          train_loader,
#     #          validation_loader,
#     #          optimizer,
#     #          criterion,
#     #          num_epoch=5)


# if __name__ == "__main__":
#     main()
