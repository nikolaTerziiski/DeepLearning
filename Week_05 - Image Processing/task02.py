from skimage import data, color
from skimage.filters import try_all_threshold, threshold_otsu, threshold_local
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def show_image(ax, image, title='Image', cmap_type='gray'):
  ax.imshow(image, cmap=cmap_type)
  ax.set_title(title)
  ax.axis('off')


def main():
    
    pil = Image.open("../DATA/w05_chess_pieces.png")
    img_chess = np.array(pil)
    img_chess_grayscale = color.rgb2gray(img_chess)
    #fig, axis = try_all_threshold(img_grayscale, verbose=False)
    
    thresh_global = threshold_otsu(img_chess_grayscale)
    binary_global = img_chess_grayscale > thresh_global
    
    fig,axes = plt.subplots(2,3, figsize=(10,5))
    
    fig.suptitle("Thresholding images")
    show_image(axes[0,0], img_chess_grayscale, 'Original')
    show_image(axes[0,1], binary_global, 'Global thresholding')
    
    
    thresh_local = threshold_local(img_chess_grayscale, block_size=511, offset=0.01)
    binary_local = img_chess_grayscale > thresh_local
    
    show_image(axes[0,2], binary_local, "Local Thresholding")
    
    
    #Text image:
    pil = Image.open("../DATA/w05_text_page.png")
    img_text = np.array(pil)
    img_text_grayscale = color.rgb2gray(img_text)
    thresh_text_global = threshold_otsu(img_text_grayscale)
    binary_text_global = img_text_grayscale > thresh_text_global
    
    show_image(axes[1,0], img_text_grayscale, "Original")
    show_image(axes[1,1], binary_text_global, 'Global thresholding')
    
    
    thresh_text_local = threshold_local(img_text_grayscale, block_size=35, offset=0.01)
    binary_text_local = img_text_grayscale > thresh_text_local
    show_image(axes[1,2], binary_text_local, 'Local tresholding')
    plt.show()
    
    pil = Image.open("../DATA/w05_fruits.png")
    img_fruits = np.array(pil)
    img_fruits_grayscale = color.rgb2gray(img_fruits)
    
    fix,axis = try_all_threshold(img_fruits_grayscale)
    plt.show()
    #I think based on all 8, the best one is the Li algorithm. I can see clear the berries. For more of the others they are literally combined
    
    #Shapes
    pil = Image.open("../DATA/w05_shapes.png")
    img_shapes = np.array(pil)
    img_shapes_grayscale = color.rgb2gray(img_shapes)
    
    thresh_shapes = threshold_otsu(img_shapes_grayscale)
    binary_global_shapes = img_shapes_grayscale > thresh_shapes
    
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    show_image(axes[0], img_shapes)
    show_image(axes[1], binary_global_shapes)
    
    plt.show()
    return
    
if __name__ == "__main__":
    main()