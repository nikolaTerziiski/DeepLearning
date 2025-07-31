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
    return np.random.uniform(start, finish)


if __name__ == '__main__':
    dataset = create_dataset(6)
    w = initialize_weights(0, 10)
    
    learning_rate = 0.001
    eps = 0.001
    for epoch in range(500):
        loss = calculate_loss(w, dataset)

        print("Loss function before: ", loss)
        new_w = w + learning_rate
        new_loss = calculate_loss(new_w, dataset)
        print("Loss function after: ", new_loss)
        
        grad = (new_loss - loss) / eps
        
        w = w - learning_rate * grad
