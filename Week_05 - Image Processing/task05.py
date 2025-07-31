from skimage import data, color
from skimage.filters import sobel, gaussian, threshold_otsu, threshold_local, try_all_threshold
from skimage.restoration import inpaint_biharmonic
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def show_image(ax, image, title='Image', cmap_type='gray'):
  ax.imshow(image, cmap=cmap_type)
  ax.set_title(title)
  ax.axis('off')
  
  
def main():
    
    pil_astro = Image.open("../DATA/w05_damaged_astro.png")
    pil_logo = Image.open("../DATA/w05_logo_image.png")
    
    img_astro = np.array(pil_astro)
    img_logo = np.array(pil_logo)
    
    mask_astro = np.zeros(img_astro.shape[:-1], dtype=bool)
    mask_astro[165:190, 70:165] = 1
    mask_astro[365:415, 75:100] = 1
    
    img_logo_grayscale = color.rgb2gray(img_logo)
    
    mask_logo = np.zeros(img_logo.shape[:2], dtype=bool)
    mask_logo[200:270, 340:420] = True
    
    fig,axes = plt.subplots(2,2, figsize=(10,5))
    show_image(axes[0,0], img_astro, 'Original')
    show_image(axes[0,1], inpaint_biharmonic(img_astro, mask_astro, channel_axis=2))
    show_image(axes[1,0], img_logo, 'Original')
    show_image(axes[1,1], inpaint_biharmonic(img_logo, mask_logo, channel_axis=2))
    plt.show()
    
    return


if __name__ == "__main__":
    main()