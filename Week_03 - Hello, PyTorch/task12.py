import random
import torch
import torch.nn as nn
import numpy as np


def main():
    
    #Basically I create a tensor with 1 element, I take if rom the uniform distribution and then taking the value with .item()
    params = [(nn.init.uniform_(torch.empty(1), 0.0001, 0.01).item(),nn.init.uniform_(torch.empty(1), 0.85, 0.95).item()) for _ in range(10)]

    print(params)


if __name__ == "__main__":
    main()
