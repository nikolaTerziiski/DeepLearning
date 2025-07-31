import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.nn import CrossEntropyLoss


def main():

    arr = np.random.uniform(low=0.0, high=1.0, size=(12, 9))

    samples = arr[:, :-1]
    tensor_samples = torch.tensor(samples, dtype=torch.float64)
    labels = arr[:, -1:]
    tensor_labels = torch.tensor(labels, dtype=torch.float64)

    dataset = TensorDataset(tensor_samples, tensor_labels)

    last_sample, last_label = dataset[-1]
    print(f'Last sample: {last_sample}')

    print(f'Last label: {last_label}')


if __name__ == "__main__":
    main()
