import numpy as np

def show_numpy_array():
    baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]
    
    np_baseball = np.array(baseball)
    print(f"Type: {type(np_baseball)}")
    print(f"Number of rows and columns: ({np_baseball[:,0].size}, {np_baseball[0,:].size})")
    
if __name__ == '__main__':
    show_numpy_array()