from PIL import Image
from skimage import data, segmentation, color
from skimage.feature import canny, Cascade
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def show_image(ax, image, title="Original", cmap_type="gray"):
    ax.imshow(image, cmap=cmap_type)
    ax.set_title(title)
    ax.axis('off')
    
    
def main():
    person_night_img = np.array(Image.open("../DATA/w05_person_at_night.jpg"))
    friends_img = np.array(Image.open("../DATA/w05_friends.jpg"))
    lady_img = np.array(Image.open("../DATA/w05_lady.jpg"))
    
    
    trained_file = data.lbp_frontal_face_cascade_filename()
    detector = Cascade(xml_file=trained_file)
    
    
    detected = detector.detect_multi_scale(img=person_night_img, scale_factor=1.2, step_ratio=1, min_size=(10, 10), max_size=(200, 200))
    
    fig, axes = plt.subplots(3,9, figsize=(15,5))
    show_image(axes[0,0], person_night_img, 'Original')
    show_image(axes[0,1], person_night_img, 'Face detected')
    axes[0][3].set_visible(False)
    axes[0][4].set_visible(False)
    axes[0][5].set_visible(False)
    axes[0][6].set_visible(False)
    axes[0][7].set_visible(False)
    axes[0][8].set_visible(False)
    
    man_coordinates = detected[0]
    man_face_img = person_night_img[man_coordinates['r']: man_coordinates['r'] + man_coordinates['height'], man_coordinates['c']: man_coordinates['c']+man_coordinates['width']]
    
    show_image(axes[0,2], man_face_img, 'Face Image')
    for patch in detected:
        axes[0,1].axes.add_patch(
            patches.Rectangle(
                (patch['c'], patch['r']),
                patch['width'],
                patch['height'],
                fill=False,
                color='r',
                linewidth=1,
            )
        )

    detected = detector.detect_multi_scale(img=friends_img, scale_factor=1.2, step_ratio=1, min_size=(10, 10), max_size=(200, 200))
    
    show_image(axes[1,0], friends_img, 'Original')
    show_image(axes[1,1], friends_img, 'Face detected')
    for patch in detected:
        axes[1,1].axes.add_patch(
            patches.Rectangle(
                (patch['c'], patch['r']),
                patch['width'],
                patch['height'],
                fill=False,
                color='r',
                linewidth=1,
            )
        )
    counter=2
    for detection in detected:
        detected_face = friends_img[detection['r']: detection['r'] + detection['height'], detection['c']: detection['c']+detection['width']]
        show_image(axes[1,counter], detected_face, f'Person {counter}')
        counter += 1

    lady_segments = segmentation.slic(lady_img, n_segments=400)
    segmented_lady_image = color.label2rgb(lady_segments, lady_img,kind='avg')
    detected = detector.detect_multi_scale(img=lady_img, scale_factor=1.2, step_ratio=1, min_size=(10, 10), max_size=(200, 200))
    lady_coordinates = detected[0]
    lady_face_img = segmented_lady_image[lady_coordinates['r']: lady_coordinates['r'] + lady_coordinates['height'], lady_coordinates['c']: lady_coordinates['c']+lady_coordinates['width']]

    show_image(axes[2,0], lady_img, 'Original')
    show_image(axes[2,1], segmented_lady_image, 'Face image')
    show_image(axes[2,2], lady_face_img, 'Face detected')
    axes[2][3].set_visible(False)
    axes[2][4].set_visible(False)
    axes[2][5].set_visible(False)
    axes[2][6].set_visible(False)
    axes[2][7].set_visible(False)
    axes[2][8].set_visible(False)
    
    for patch in detected:
        axes[2,1].axes.add_patch(
            patches.Rectangle(
                (patch['c'], patch['r']),
                patch['width'],
                patch['height'],
                fill=False,
                color='r',
                linewidth=1,
            )
        )
    plt.show()
    return



if __name__ == "__main__":
    main()