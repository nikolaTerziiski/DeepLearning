import numpy as np


def create_dataset(n):

    list = []
    for i in range(n):
        list.append((i, i * 2))

    return list

def calculate_loss(w, dataset):
    #I saw the formula 
    errors = [(w * x - y)**2 for x, y in dataset]
    return np.mean(errors)

def initialize_weights(start, finish):
    np.random.seed(42) 
    return np.random.uniform(start, finish)


if __name__ == '__main__':
    dataset = create_dataset(6)
    w = initialize_weights(0,10)
    loss = calculate_loss(w, dataset)
    print(f'MSE: {loss}')
    
    #So the w + 0.001 * 2, w + 0.001, w - 0.001 and w - 0.001 * 2 depends. 
    #for example if w is under 2 then w - 0.001 * 2 will be negative and the loss will be high.
    #if w is over 2 then w - 0.001 * 2 will be positive and the loss will be low and vise versa for the other values

