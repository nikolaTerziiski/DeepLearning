import numpy as np
import matplotlib.pyplot as plt


def show_numpy_array():

    np.random.seed(123)

    all_walks = []
    for i in range(5):
        floor_level = 0
        steps_chronology = []
        throw_real_dice = np.random.randint(1, 7)
        for k in range(100):

            if np.random.random() < 0.005:
                floorLevel = 0
                steps_chronology.append(floorLevel)
                continue

            steps_chronology.append(floor_level)
            if throw_real_dice in {1, 2}:
                if floor_level > 0:
                    floor_level -= 1
            elif throw_real_dice in {3, 4, 5}:
                floor_level += 1
            else:
                value_to_add = np.random.randint(1, 7)
                floor_level += value_to_add
            throw_real_dice = np.random.randint(1, 7)
        all_walks.append(steps_chronology)

    numpy_arr = np.array(all_walks)
    end_steps = numpy_arr[:, -1]

    plt.hist(end_steps, bins=10)
    plt.show()


if __name__ == '__main__':
    show_numpy_array()
