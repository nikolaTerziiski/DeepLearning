import numpy as np

dataset_or = [(0,0,0), (0,1,1), (1,0,1), (1,1,1)]
dataset_nand = [(0,0,1), (0,1,1), (1,0,1), (1,1,0)]
dataset_xor = [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_loss(w1, w2, b, dataset):
    errors = [(sigmoid(w1 * x1 + w2 * x2 + b) - y)**2 for x1, x2, y in dataset]
    return np.mean(errors)

def initialize_weights(start=0, end=1):
    return np.random.uniform(start, end)

def train_neuron(dataset, label, epochs=100000, learning_rate=0.01, eps=0.001):
    w1 = initialize_weights()
    w2 = initialize_weights()
    b = initialize_weights()

    for epoch in range(epochs):
        loss = calculate_loss(w1, w2, b, dataset)

        grad_w1 = (calculate_loss(w1 + eps, w2, b, dataset) - loss) / eps
        grad_w2 = (calculate_loss(w1, w2 + eps, b, dataset) - loss) / eps
        grad_b  = (calculate_loss(w1, w2, b + eps, dataset) - loss) / eps

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        b  -= learning_rate * grad_b

        if epoch % 10000 == 0:
            print(f"{label} | Epoch {epoch}, Loss: {loss:.6f}")

    return w1, w2, b

class Xor:
    def __init__(self, w_or1, w_or2, b_or, w_nand1, w_nand2, b_nand, w_and1, w_and2, b_and):
        self.w_or1 = w_or1
        self.w_or2 = w_or2
        self.b_or = b_or
        self.w_nand1 = w_nand1
        self.w_nand2 = w_nand2
        self.b_nand = b_nand
        self.w_and1 = w_and1
        self.w_and2 = w_and2
        self.b_and = b_and

    def forward(self, x1, x2):
        out_or = sigmoid(self.w_or1 * x1 + self.w_or2 * x2 + self.b_or)
        out_nand = sigmoid(self.w_nand1 * x1 + self.w_nand2 * x2 + self.b_nand)
        out_xor = sigmoid(self.w_and1 * out_or + self.w_and2 * out_nand + self.b_and)
        return out_xor

if __name__ == '__main__':
    
    
    w_or1, w_or2, b_or = train_neuron(dataset_or, "OR")
    
    w_nand1, w_nand2, b_nand = train_neuron(dataset_nand, "NAND")
    
    dataset_final = []

    for x1, x2, y in dataset_xor:
        out_or = sigmoid(w_or1 * x1 + w_or2 * x2 + b_or)
        out_nand = sigmoid(w_nand1 * x1 + w_nand2 * x2 + b_nand)
        dataset_final.append((out_or, out_nand, y))

    w_and1, w_and2, b_and = train_neuron(dataset_final, "XOR")

    xor_model = Xor(
        w_or1, w_or2, b_or,
        w_nand1, w_nand2, b_nand,
        w_and1, w_and2, b_and
    )

    for x1, x2, y in dataset_xor:
        output = xor_model.forward(x1, x2)
        print(f"Input: ({x1}, {x2}) => Predicted: {output:.4f}, Rounded: {round(output)}, Actual: {y}")
