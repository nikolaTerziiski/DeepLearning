import os
import re
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from collections import Counter
import pandas as pd

# Определение на OmniDataset (как е дадено от потребителя)
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

                        # Запазваме пътя до файла, името на азбуката и етикета (1-базиран)
                        self.samples.append(
                            (img_path.path, alphabet, character_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, alphabet_name, character_label = self.samples[idx]
        one_hot = torch.zeros(self.num_alphabets, dtype=torch.float32)
        img = Image.open(image_path).convert('L')
        img = self.transform(img)
        alpha_idx = self.alphabet_to_idx[alphabet_name]
        one_hot[alpha_idx] = 1.0
        # Връщаме 0-базиран етикет
        return img, one_hot, (character_label - 1)

# Функция за изчисляване на разпределението на етикетите чрез DataLoader
def compute_label_distribution_via_dataloader(root_path, transform, batch_size=512):
    dataset = OmniDataset(root_path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    counter = Counter()
    for imgs, one_hot_alpha, labels in loader:
        for lbl in labels.tolist():
            counter[lbl] += 1
    return counter

# Основна функция
def main():
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    # Път към тренировъчния сет
    train_path = "../DATA/omniglot_train"
    # Изчисляваме разпределението на етикетите
    label_counter = compute_label_distribution_via_dataloader(train_path, transformer)

    # Преобразуваме Counter в pandas DataFrame
    df = pd.DataFrame({
        "Label": list(label_counter.keys()),
        "Count": list(label_counter.values())
    })
    df = df.sort_values(by="Label").reset_index(drop=True)

    # Показваме DataFrame на потребителя
    import ace_tools as tools; tools.display_dataframe_to_user(name="Label Distribution", dataframe=df)

if __name__ == "__main__":
    main()
