import torch
import numpy as np
def main():
    xs = [[1, 2, 3], [4, 5, 6]]
    tensor = torch.tensor(xs)
    print(type(tensor))
    print(tensor.device)
    
if __name__ == "__main__":
    main()