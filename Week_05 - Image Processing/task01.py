from skimage import data, color
import matplotlib.pyplot as plt
import numpy as np

def main():
    rocket = data.rocket()
    grayscale = color.rgb2gray(rocket)
    
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    
    axes[0, 0].imshow(rocket)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(grayscale, cmap='gray')
    axes[0, 1].set_title("Grayscaled")
    axes[1, 0].imshow(np.fliplr(rocket))
    axes[1, 0].set_title("Horizotnal Flip")
    axes[1, 1].imshow(np.flipud(rocket))
    axes[1, 1].set_title("Vertical flip")
       
    plt.show()
    
    red_channel_rgb = rocket[:,:,0]
    green_channel_rgb = rocket[:,:,1]
    blue_channel_rgb = rocket[:,:,2]
    
    fig,axes = plt.subplots(3,3, figsize=(17,11))
    
    fig.suptitle("Playing with intensities")
    
    axes[0,0].set_title('Red')
    axes[0,0].imshow(red_channel_rgb)
    
    axes[0,1].set_title('Green')
    axes[0,1].imshow(green_channel_rgb)
    
    axes[0,2].set_title('Blue')
    axes[0,2].imshow(blue_channel_rgb)
    
    axes[1,0].set_title('Red')
    axes[1,0].imshow(red_channel_rgb, cmap='gray')
    
    axes[1,1].set_title('Green')
    axes[1,1].imshow(green_channel_rgb, cmap='gray')
    
    axes[1,2].set_title('Blue')
    axes[1,2].imshow(blue_channel_rgb, cmap='gray')
    
    axes[2,0].hist(red_channel_rgb.flatten(), bins=256)
    axes[2,0].set_title('Red pixle distribution')
    axes[2,0].set_xlabel('Intensity')
    axes[2,0].set_ylabel('Number of pixels')
    
    axes[2,1].hist(green_channel_rgb.flatten(), bins=256)
    axes[2,1].set_title('Green pixle distribution')
    axes[2,1].set_xlabel('Intensity')
    axes[2,1].set_ylabel('Number of pixels')
    
    axes[2,2].hist(blue_channel_rgb.flatten(), bins=256)
    axes[2,2].set_title('Blue pixle distribution')
    axes[2,2].set_xlabel('Intensity')
    axes[2,2].set_ylabel('Number of pixels')

    plt.show()
    
if __name__ == '__main__':
    main()