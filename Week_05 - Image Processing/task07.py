from PIL import Image
from skimage import data, color
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt


def show_image(ax, image, title="Original", cmap_type="gray"):
    ax.imshow(image, cmap=cmap_type)
    ax.set_title(title)
    ax.axis('off')
    
    
def main():
    
    horse_img = data.horse()
    erosed_horse = exposure.equalize_hist(horse_img)
    threshold_horse = threshold_otsu(erosed_horse)
    binary_horse_global = erosed_horse <= threshold_horse
    
    countours_horse = find_contours(erosed_horse)
    
    
    #-- dice
    dice_img_original = np.array(Image.open("../DATA/w05_dice.png").convert("RGB"))
    dice_img_grayscaled = color.rgb2gray(dice_img_original)
    
    filtered_dice = exposure.equalize_adapthist(dice_img_grayscaled, clip_limit=0.03)
    
    threshold_img_dice = threshold_otsu(filtered_dice)
    binary_dice_global = filtered_dice <= threshold_img_dice
    
    fig,axes = plt.subplots(2,2, figsize=(10,5))
    
    contours = find_contours(binary_dice_global)
    show_image(axes[0,0], dice_img_original, 'Original')
    show_image(axes[0,1], binary_dice_global, 'Thresh')
    
    dot_counter = [c for c in contours if len(c) < 50]
    print(f"Dots count: {len(dot_counter)}")
    for contour in contours:
        axes[0,1].plot(contour[:, 1], contour[:, 0], linewidth=2)
    
    
    show_image(axes[1,0], horse_img, 'Original')
    show_image(axes[1,1], binary_horse_global, 'Thresh')
    for contour in countours_horse:
        axes[1,1].plot(contour[:, 1], contour[:, 0], linewidth=2)
    
    plt.show()
    return


if __name__ == "__main__":
    main()