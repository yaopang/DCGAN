import pickle
import cv2
import os
import h5py

def get_images_and_labels(image_dir, labels_file):
    images = get_images(image_dir)
    labels_dict = get_labels(labels_file)


    dataset = {}
    dataset.update(labels_dict)
    dataset.update({'images': images})
    with open('image_dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)
    pass


def get_labels(labels_file):
    with open(labels_file, 'rb') as f:
        labels = []
        filenames = []
        for i, line in enumerate(f):
            if i != 0 and i != 1:
                filenames.append(line.split()[0])
                labels.append(line.split()[20])
        labels_dict = {'filenames': filenames, 'labels': labels}
        return labels_dict


def get_images(directory):
    filenames = os.listdir(directory)
    filenames = sorted(filenames)
    images = [cv2.cvtColor(cv2.imread(os.path.join(directory, filename)), cv2.COLOR_BGR2RGB) for filename in filenames]
    return images


if __name__ == "__main__":
    get_images_and_labels(image_dir='data', labels_file='list_attr_celeba.txt')