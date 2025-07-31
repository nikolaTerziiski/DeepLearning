from PIL import Image
from skimage import color
from skimage.feature import canny
from skimage.feature import corner_harris, corner_peaks
import numpy as np
import matplotlib.pyplot as plt


def show_image(ax, image, title="Original", cmap_type="gray"):
    ax.imshow(image, cmap=cmap_type)
    ax.set_title(title)
    ax.axis('off')

def main():
    
    grapefruit_img = np.array(Image.open("../DATA/w05_grapefruit.jpg"))
    building_img = np.array(Image.open("../DATA/w05_single_building.jpg"))
    
    
    grapefruit_img_grayscale = color.rgb2gray(grapefruit_img)
    building_img_grayscale = color.rgb2gray(building_img)
    
    fig, axes = plt.subplots(2,5, figsize=(10,10))
    
    show_image(axes[0,0], grapefruit_img, "Original")
    show_image(axes[0,1], canny(grapefruit_img_grayscale, sigma=0.8), 'Sigma 0.8')
    show_image(axes[0,2], canny(grapefruit_img_grayscale, sigma=1), 'Sigma 1')
    show_image(axes[0,3], canny(grapefruit_img_grayscale, sigma=1.8), 'Sigma 1.8')
    show_image(axes[0,4], canny(grapefruit_img_grayscale, sigma=2.2), 'Sigma 2.2')
    
    coords_1 = corner_peaks(corner_harris(building_img_grayscale), min_distance=10, threshold_rel=0.01)
    coords_2 = corner_peaks(corner_harris(building_img_grayscale), min_distance=10, threshold_rel=0.02)
    coords_3 = corner_peaks(corner_harris(building_img_grayscale), min_distance=20, threshold_rel=0.03)
    coords_4 = corner_peaks(corner_harris(building_img_grayscale), min_distance=60, threshold_rel=0.02)
    
    print(f'With min_distance=10 and treshold_rel=0.01 we detected a total of {len(coords_1)} corners in the image')
    print(f'With min_distance=10 and treshold_rel=0.02 we detected a total of {len(coords_2)} corners in the image')
    print(f'With min_distance=20 and treshold_rel=0.03 we detected a total of {len(coords_3)} corners in the image')
    print(f'With min_distance=60 and treshold_rel=0.02 we detected a total of {len(coords_4)} corners in the image')

    show_image(axes[1,0], building_img, "Original")
    show_image(axes[1,1], building_img, 'min_distance=10 | treshold_rel=0.01')
    show_image(axes[1,2], building_img, 'min_distance=10 | treshold_rel=0.02')
    show_image(axes[1,3], building_img, 'min_distance=20 | treshold_rel=0.03')
    show_image(axes[1,4], building_img, 'min_distance=60 | treshold_rel=0.02')
    
    for i in coords_1:
        axes[1,1].plot(coords_1[:, 1], coords_1[:, 0], '+r', markersize=10)
    for i in coords_2:
        axes[1,2].plot(coords_2[:, 1], coords_2[:, 0], '+r', markersize=10)
    for i in coords_3:
        axes[1,3].plot(coords_3[:, 1], coords_3[:, 0], '+r', markersize=10)
    for i in coords_4:
        axes[1,4].plot(coords_4[:, 1], coords_4[:, 0], '+r', markersize=10) 
        
    plt.show()
    return


if __name__ == "__main__":
    main()