import numpy as np
import torch.nn as nn


def main():
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 16), nn.Linear(16, 6),
                          nn.Linear(6, 1))

    for i, layer in enumerate(model):
        nn.init.uniform_(layer.weight, 0, 1)
        nn.init.uniform_(layer.bias, 0, 1)
        if i < 2:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False

    for i, layer in enumerate(model):
        print(f'Layer {1+i}:')
        print(layer.weight[:5])
        print(layer.bias[:5])


if __name__ == "__main__":
    main()
