from PIL import Image
from skimage import data, segmentation, color, transform
from skimage.feature import canny, Cascade
from skimage.filters import gaussian
from skimage.restoration import inpaint_biharmonic
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def show_image(ax, image, title="Original", cmap_type="gray"):
    ax.imshow(image, cmap=cmap_type)
    ax.set_title(title)
    ax.axis('off')


def main():
    
    music_group_img = np.array(Image.open("../DATA/w05_music_group.jpg"))
    graduation_img = np.array(Image.open("../DATA/w05_graduation.jpg"))
    
    trained_file = data.lbp_frontal_face_cascade_filename()
    detector = Cascade(xml_file=trained_file)
    
    detected = detector.detect_multi_scale(img=music_group_img, scale_factor=1.2, step_ratio=1, min_size=(10, 10), max_size=(200, 200))
    
    fig,axes = plt.subplots(2,2, figsize=(10,5))
    show_image(axes[0,0], music_group_img, 'Original')
    
    for detection in detected:
        detected_face = music_group_img[detection['r']: detection['r'] + detection['height'], detection['c']: detection['c']+detection['width']]
        
        blurred_face = gaussian(detected_face, sigma=10, channel_axis=2)
        blurred_face = (blurred_face * 255).astype(np.uint8)
        music_group_img[detection['r']: detection['r'] + detection['height'], detection['c']: detection['c']+detection['width']] = blurred_face
    
    show_image(axes[0,1], music_group_img, 'Edited')
    
    show_image(axes[1,0], graduation_img, 'Original')
    rotated = transform.rotate(graduation_img, angle=20, resize=False)
    denoised_image = gaussian(rotated, channel_axis=2)
    denoised_image = (denoised_image * 255).astype(np.uint8)
    
    mask_graduation = np.zeros(denoised_image.shape[:-1], dtype=bool)
    mask_graduation[315:360, 135:180] = 1
    mask_graduation[445:480, 465:500] = 1
    mask_graduation[130:150, 350:370] = 1
    show_image(axes[1,1], inpaint_biharmonic(denoised_image, mask_graduation, channel_axis=2), 'Fixed')
    plt.show()
    return


if __name__ == "__main__":
    main()