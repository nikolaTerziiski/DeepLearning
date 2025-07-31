import torch
import torch.nn as nn


def main():

    temperature_observation = [3, 4, 6, 2, 3, 6, 8, 9]

    tensor = torch.tensor(temperature_observation,
                          dtype=torch.float32).unsqueeze(0)

    model = nn.Sequential(nn.Linear(in_features=8, out_features=1))

    logit = model(tensor)
    print(logit[0][0].item())

    #None of them are false


if __name__ == "__main__":
    main()
