import torch
import numpy as np
import torch.nn as nn


def main():

    first_model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 4),
                                nn.Linear(4, 1))

    second_model = nn.Sequential(nn.Linear(15, 10), nn.Linear(10, 5),
                                 nn.Linear(5, 4), nn.Linear(4, 1))

    print(
        f'Number of parameters in network 1: {sum(param.numel() for param in first_model.parameters())}'
    )
    print(
        f'Number of parameters in network 2: {sum(param.numel() for param in second_model.parameters())}'
    )


if __name__ == "__main__":
    main()
