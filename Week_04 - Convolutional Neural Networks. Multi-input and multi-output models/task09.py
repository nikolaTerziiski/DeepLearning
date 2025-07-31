from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import torch as torch
import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import re


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


def main():
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((64, 64))])

    dataset = OmniDataset("../DATA/omniglot_train", train_transform)

    # new_dataset = ImageFolder("../DATA/omniglot_train", train_transform)
    # print(new_dataset.classes[0].classes)
    length_dataset = dataset.__len__()
    print(f"Length: {length_dataset}")
    image, index, neshto = dataset.__getitem__(length_dataset - 1)
    print((image, index, neshto))
    print(f"Shape of the last image: {image.shape}")


if __name__ == '__main__':
    main()
