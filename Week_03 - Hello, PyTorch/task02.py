import torch
import torch.nn as nn


def main():

    temperature_observation = [2, 3, 6, 7, 9, 3, 2, 1]

    tensor = torch.tensor(temperature_observation,
                          dtype=torch.float32).unsqueeze(0)

    model = nn.Sequential(nn.Linear(in_features=8, out_features=4),
                          nn.Linear(in_features=4, out_features=1))

    logit = model(tensor)
    print(logit)


if __name__ == "__main__":
    main()
