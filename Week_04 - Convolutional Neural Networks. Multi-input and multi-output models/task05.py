from PIL import Image
from torchvision import transforms 
from torch.utils import data
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import numpy as np

def main():
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128))
    ])
    
    dataset_train = ImageFolder(
        '../DATA/clouds/clouds_train',
        transform=train_transform
    )
    
    dataloader_train = data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=1,
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle("The clouds dataset")
    axes = axes.flatten()
    for i in range(6):
        index = np.random.randint(0, len(dataset_train))
        image, label = dataset_train[index]
        image = image.permute(1, 2, 0).numpy()
        axes[i].imshow(image)
        axes[i].set_title(f"{dataset_train.classes[label]}")
        axes[i].axis('off')

    plt.show()

if __name__ == "__main__":
    main()
    