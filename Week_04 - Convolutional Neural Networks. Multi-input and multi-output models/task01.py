import pandas as pd
import torch
import numpy as np


class WaterDataset:

    def __init__(self, path_to_csv):
        super().__init__()

        self.data = pd.read_csv(path_to_csv).to_numpy()
        self.features = self.data[:, :-1]
        self.target = self.data[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.features[index], self.target[index].item())


def main():
    return


if __name__ == "__main__":

    dataset = WaterDataset('../DATA/water_train.csv')
    print(f"Number of instance: {len(dataset)}")
    print(f"Fifth item: {dataset[4]}")

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=2,
                                             shuffle=True)
    print(f"{next(iter(dataloader))}")

    #batch_features, batch_target = next(iter(dataloader))
    main()
