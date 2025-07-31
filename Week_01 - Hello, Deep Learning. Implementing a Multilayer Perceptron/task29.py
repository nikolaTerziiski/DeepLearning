import numpy as np


def main(n):

    result = [(i, i*2) for i in range(n)]
    print(result)
    print(initialize_weights(0, 100))
    print(initialize_weights(0, 10))


def initialize_weights(start, finish):
    return np.random.uniform(start, finish)


if __name__ == '__main__':
    main(4)
