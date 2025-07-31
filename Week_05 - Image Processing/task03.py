from skimage import data, color
from skimage.filters import sobel, gaussian
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def show_image(ax, image, title='Image', cmap_type='gray'):
  ax.imshow(image, cmap=cmap_type)
  ax.set_title(title)
  ax.axis('off')


def main():
    
    soap_pil = Image.open('../DATA/w05_soaps.jpg')
    building_pil = Image.open('../DATA/w05_building_image.jpg')
    xray_pil = Image.open('../DATA/w05_xray.png')
    aerial_pil = Image.open('../DATA/w05_aerial.png')
    
    soap_img = np.array(soap_pil)
    building_img = np.array(building_pil)
    xray_img = np.array(xray_pil)
    aerial_img = np.array(aerial_pil)
    coffee_img = data.coffee()
    
    soap_grayscale = color.rgb2gray(soap_img)
    xray_grayscale = np.array(xray_img)
    aerial_grayscale = np.array(aerial_img)
    
    soap_sobel = sobel(soap_grayscale)
    building_gaussian = gaussian(building_img)
    xray_standard_equalization = exposure.equalize_hist(xray_img)
    aerial_standard_equalization = exposure.equalize_hist(aerial_img)
    coffee_standard_equalizatiion = exposure.equalize_adapthist(coffee_img, clip_limit=0.03)
    
    fig, axes = plt.subplots(5,2, figsize=(10,5))
    
    show_image(axes[0,0], soap_img, 'Original')
    show_image(axes[0,1], soap_sobel, 'Sobeled')
    show_image(axes[1,0], building_img, 'Original')
    show_image(axes[1,1], building_gaussian, 'Image blurring')
    show_image(axes[2,0], xray_img, 'Original')
    show_image(axes[2,1], xray_standard_equalization, 'Standard histogram equalization')
    show_image(axes[3,0], aerial_img, 'Original')
    show_image(axes[3,1], aerial_standard_equalization, 'Standard histogram equalization')
    show_image(axes[4,0], coffee_img, 'Original')
    show_image(axes[4,1], coffee_standard_equalizatiion, 'Standard histogram equalization')
    
    
    plt.show()
    
    return
    
if __name__ == "__main__":
    main()