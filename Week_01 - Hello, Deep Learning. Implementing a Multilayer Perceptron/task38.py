import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
dataset_for_xor = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Xor:
    def __init__(self):
        self.w1 = np.random.uniform(0, 1)
        self.w2 = np.random.uniform(0, 1)
        self.b = np.random.uniform(0, 1)

    def forward(self, x1, x2):
        return sigmoid(self.w1 * x1 + self.w2 * x2 + self.b)

    def calculate_loss(self, dataset):
        
        errors = [(self.forward(x1, x2) - y) ** 2 for x1, x2, y in dataset]
        return np.mean(errors)


if __name__ == '__main__':
    model = Xor()

    learning_rate = 0.001
    eps = 0.001
    epochs = 100000
    losses = []

    for epoch in range(epochs):
        loss = model.calculate_loss(dataset_for_xor)
        losses.append(loss)

        new_w1 = model.w1 + eps
        new_w2 = model.w2 + eps
        new_b = model.b + eps

        grad_w1 = (Xor().calculate_loss([(x1, x2, y) for x1, x2, y in dataset_for_xor]) - loss) / eps
        grad_w2 = (Xor().calculate_loss([(x1, x2, y) for x1, x2, y in dataset_for_xor]) - loss) / eps
        grad_b = (Xor().calculate_loss([(x1, x2, y) for x1, x2, y in dataset_for_xor]) - loss) / eps

        model.w1 -= learning_rate * grad_w1
        model.w2 -= learning_rate * grad_w2
        model.b -= learning_rate * grad_b

        if epoch % 5000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}, w1: {model.w1:.4f}, w2: {model.w2:.4f}, bias: {model.b:.4f}")

    print("\nTesting model on XOR dataset:")
    for x1, x2, y in dataset_for_xor:
        pred = model.forward(x1, x2)
        print(f"Input: ({x1}, {x2}) => Predicted: {pred:.4f}, Actual: {y}")

    plt.plot(losses)
    plt.title("Loss over Epochs for XOR")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
