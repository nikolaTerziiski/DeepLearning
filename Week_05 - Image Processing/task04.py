from skimage import data, color
from skimage.filters import sobel, gaussian, threshold_otsu, threshold_local, try_all_threshold
from skimage.restoration import denoise_bilateral
from skimage.morphology import binary_dilation, binary_erosion
from skimage import exposure
from skimage.feature import canny
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def show_image(ax, image, title='Image', cmap_type='gray'):
  ax.imshow(image, cmap=cmap_type)
  ax.set_title(title)
  ax.axis('off')
  
  
def main():
    
    
    r_pil = Image.open("../DATA/w05_r5.png")
    continents_pil = Image.open("../DATA/w05_continents.jpg")
    
    r_img = np.array(r_pil)
    r_img_grayscaled = color.rgb2gray(r_img)
    
    continents_img = np.array(continents_pil)
    
    fig,axes = plt.subplots(2,2, figsize=(10,5))
    
    show_image(axes[0,0], r_img,'Original')
    show_image(axes[0,1], binary_erosion(r_img_grayscaled), 'Noise removed')
    show_image(axes[1,0], continents_img,'Original')
    show_image(axes[1,1], binary_dilation(continents_img), 'Noise removed')
    
    plt.show()
    
    return


if __name__ == "__main__":
    main()