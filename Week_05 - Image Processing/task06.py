from skimage import data, color, segmentation
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def show_image(ax, image, title='Image', cmap_type='gray'):
  ax.imshow(image, cmap=cmap_type)
  ax.set_title(title)
  ax.axis('off')
  
def main():
    
    fruit_img = np.array(Image.open("../DATA/w05_fruits_generic.png"))
    dog_img = np.array(Image.open("../DATA/w05_miny.png"))
    landscape_img = np.array(Image.open("../DATA/w05_landscape.jpg"))
    lady_img = np.array(Image.open("../DATA/w05_lady.jpg"))
    
    fig,axes = plt.subplots(4,2, figsize=(10,5))
    
    lady_segments = segmentation.slic(lady_img, n_segments=400)
    segmented_lady_image = color.label2rgb(lady_segments, lady_img,kind='avg')
    
    show_image(axes[0,0], fruit_img, 'Original')
    show_image(axes[0,1], random_noise(fruit_img), 'Noised image')
    show_image(axes[1,0], dog_img, 'Original')
    show_image(axes[1,1], gaussian(dog_img, channel_axis=2), 'Denoised image')
    show_image(axes[2,0], landscape_img, 'Original')
    show_image(axes[2,1], denoise_bilateral(landscape_img, channel_axis=2), 'Denoised image')
    show_image(axes[3,0], lady_img, 'Original')
    show_image(axes[3,1], gaussian(segmented_lady_image, channel_axis=2), 'Denoised image')
    
    plt.show()
    return


if __name__ == "__main__":
    main()