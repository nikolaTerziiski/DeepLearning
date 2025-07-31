import torch


def main():
    list = [[72, 75, 78], [70, 73, 76]]

    tensor = torch.tensor(list)
    print(f'Temperature: {tensor}')
    print(f'Shape of temperatures: {tensor.shape}')
    print(f'Data type of temperatures: {tensor.dtype}')

    tensor += 2

    print(f'Corrected temperatures: {tensor}')


if __name__ == "__main__":
    main()
