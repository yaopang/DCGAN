import numpy as np
import scipy.misc
import sys


def center_crop(data, desired_shape):
    start_crop = (np.array(data.shape[:2]) - desired_shape[:2])//2
    return data[start_crop[0]:start_crop[0]+desired_shape[0],
                start_crop[1]:start_crop[1]+desired_shape[1]]


def main():
    output_size = [80, 80, 3]
    image_path = sys.argv[1]
    image_name = image_path.split('/')[-1].split('.')[0]
    original_image = scipy.misc.imread(image_path)
    smaller_dim = min(original_image.shape[0], original_image.shape[1])
    cropped_image = center_crop(original_image, (smaller_dim, smaller_dim, original_image[2]))
    resized_image = scipy.misc.imresize(cropped_image, output_size)
    scipy.misc.imsave(image_name + '_small.jpg', resized_image)


if __name__ == "__main__":
    main()
