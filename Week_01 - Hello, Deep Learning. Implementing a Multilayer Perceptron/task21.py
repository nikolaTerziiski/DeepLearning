import numpy as np

def show_numpy_array():    
    
    np.random.seed(123)
    
    ran_number = np.random.random()
    print("Random float: ",ran_number)
    
    throw_first_dice = np.random.randint(1, 7)
    print("Random integer 1:", throw_first_dice)
    throw_second_dice = np.random.randint(1, 7)
    print("Random integer 2:", throw_second_dice)
    
    floorLevel = 50
    print("Before throw step: ", floorLevel)
    
    throwRealDice = np.random.randint(1, 7)
    print("After throw step: ", throwRealDice)
    if throwRealDice in {1,2}:
        floorLevel -= 1
    elif throwRealDice in {3,4,5}:
        floorLevel += 1
    else:
        throwRealDice = np.random.randint(1, 7)
        floorLevel += throwRealDice
    print("After throw step: ", floorLevel)
    
if __name__ == '__main__':
    show_numpy_array()