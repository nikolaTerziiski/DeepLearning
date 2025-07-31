import numpy as np
import matplotlib.pyplot as plt
def show_numpy_array():    
    
    np.random.seed(123)

    floor_level = 0
    steps_chronology = []
    throw_real_dice = np.random.randint(1, 7)
    for i in range(100):
        steps_chronology.append(floor_level)
        if throw_real_dice in {1,2}:
            if floor_level > 0:
                floor_level -= 1
        elif throw_real_dice in {3,4,5}:
            floor_level += 1
        else:
            value_to_add = np.random.randint(1, 7)
            floor_level += value_to_add
        throw_real_dice = np.random.randint(1, 7)
    print(steps_chronology)
        
    
    plt.plot(np.arange(1,101), steps_chronology)
    plt.title("Random walks")
    plt.show()
    
if __name__ == '__main__':
    show_numpy_array()