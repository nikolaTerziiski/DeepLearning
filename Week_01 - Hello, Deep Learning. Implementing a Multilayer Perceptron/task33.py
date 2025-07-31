import numpy as np

dataset_for_and = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
dataset_for_or = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]


def calculate_loss(w1, w2, dataset):
    #I saw the formula
    errors = [((w1 * x1 + w2 * x2) - y)**2 for x1, x2, y in dataset]
    return np.mean(errors)


def initialize_weights(start, finish):
    return np.random.uniform(start, finish)


if __name__ == '__main__':

    w1 = initialize_weights(0, 1)
    w2 = initialize_weights(0, 1)

    learning_rate = 0.001
    eps = 0.001
    epochs = 100000
    for epoch in range(epochs):
        loss = calculate_loss(w1, w2, dataset_for_and)

        new_w1 = w1 + learning_rate
        new_w2 = w2 + learning_rate

        new_loss_w1 = calculate_loss(new_w1, w2, dataset_for_and)
        new_loss_w2 = calculate_loss(w1, new_w2, dataset_for_and)

        grad_w1 = (new_loss_w1 - loss) / eps
        grad_w2 = (new_loss_w2 - loss) / eps

        w1 = w1 - learning_rate * grad_w1
        w2 = w2 - learning_rate * grad_w2

        if epoch % 5000 == 0:
            print(f"Epoch for and: {epoch}, Loss: {loss}, w1: {w1}, w2: {w2}")
    
    
    print("\nTesting model on AND dataset:")
    for x1, x2, y in dataset_for_and:
        prediction = w1 * x1 + w2 * x2
        print(f"Input: ({x1}, {x2}) => Predicted: {prediction:.4f}, Actual: {y}")
    
    
    
    w1 = initialize_weights(0, 1)
    w2 = initialize_weights(0, 1)

    learning_rate = 0.001
    eps = 0.001
    epochs = 100000
    for epoch in range(epochs):
        loss = calculate_loss(w1, w2, dataset_for_or)

        new_w1 = w1 + learning_rate
        new_w2 = w2 + learning_rate

        new_loss_w1 = calculate_loss(new_w1, w2, dataset_for_or)
        new_loss_w2 = calculate_loss(w1, new_w2, dataset_for_or)

        grad_w1 = (new_loss_w1 - loss) / eps
        grad_w2 = (new_loss_w2 - loss) / eps

        w1 = w1 - learning_rate * grad_w1
        w2 = w2 - learning_rate * grad_w2

        if epoch % 5000 == 0:
            print(f"Epoch for or: {epoch}, Loss: {loss}, w1: {w1}, w2: {w2}")
            
    print("\nTesting model on OR dataset:")
    for x1, x2, y in dataset_for_or:
        prediction = w1 * x1 + w2 * x2
        print(f"Input: ({x1}, {x2}) => Predicted: {prediction:.4f}, Actual: {y}")