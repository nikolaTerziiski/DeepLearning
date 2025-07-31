import numpy as np

def show_numpy_array():
    baseball = [180, 215, 210, 210, 188, 176, 209, 200]
    
    numpy_arr = np.array(baseball)
    print(numpy_arr)
    print(f"Type of baseball array: {type(numpy_arr)}")
    
if __name__ == '__main__':
    show_numpy_array()