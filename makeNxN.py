import numpy as np
import scipy.misc
import os


def visualizer_shape(num_images):
    h = int(np.floor(np.sqrt(num_images)))
    w = int(np.ceil(np.sqrt(num_images)))
    assert h * w == num_images
    return [h, w]


def visualizer_merge(images, size):
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c), dtype=np.uint8)
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img


def save_samples(images, path, name):
    vis_shape = visualizer_shape(images.shape[0])
    # images = np.uint8(255.*(images + 1.)/2.)
    vis_image = visualizer_merge(images, vis_shape)
    # cv2.imwrite(os.path.join(path, name), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    scipy.misc.imsave(os.path.join(path, name), vis_image)


male_dir = os.listdir('./good/male/')
male_images = [scipy.misc.imread('./good/male/' + male_img) for male_img in male_dir]
male_images = np.array(male_images)
save_samples(male_images, './', 'male5x5.jpg')


female_dir = os.listdir('./good/female/')
female_images = [scipy.misc.imread('./good/female/' + female_img) for female_img in female_dir]
female_images = np.array(female_images)
save_samples(female_images, './', 'female5x5.jpg')

pass
