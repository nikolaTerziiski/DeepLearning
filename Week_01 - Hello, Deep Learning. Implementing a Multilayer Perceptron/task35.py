import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':

    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)
    
    plt.plot(x, y)
    plt.title("Sigmoid Function")
    plt.show()
    
   