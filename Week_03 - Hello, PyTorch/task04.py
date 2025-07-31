import torch
from torch.nn import CrossEntropyLoss


def main():

    y = [2]
    target = torch.tensor(y)
    scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]], dtype=torch.float64)

    criterion = CrossEntropyLoss()

    result = criterion(scores, target)

    print(result)


if __name__ == "__main__":
    main()
