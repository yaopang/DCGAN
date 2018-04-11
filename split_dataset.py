import tensorflow as tf
import numpy as np
import pickle
import cv2
import os


def make_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_images_and_labels(image_dir, labels_file):
    # labels_dict = get_labels(labels_file)
    females, males = get_images(image_dir)
    chunks = list(make_chunks(females, 20000))
    for i, chunk in enumerate(chunks):
        if i != 1:
            break
        print 'female dataset', i
        np_chunk = np.array(chunk)
        np.save('face_women20k_part' + str(i), np_chunk)

    chunks = list(make_chunks(males, 20000))
    for i, chunk in enumerate(chunks):
        if i != 1:
            break
        print 'male dataset', i
        np_chunk = np.array(chunk)
        np.save('face_male20k_part' + str(i), np_chunk)


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
    labels = np.load('all_labels.npy')

    females = []
    males = []
    for i, filename in enumerate(filenames):
        image = cv2.cvtColor(cv2.imread(os.path.join(directory, filename)), cv2.COLOR_BGR2RGB)
        if labels[i] == -1:
            females.append(image)
        else:
            males.append(image)
    return females, males


if __name__ == "__main__":
    get_images_and_labels(image_dir='celebA', labels_file='list_attr_celeba.txt')
    # x = np.load('face_dataset_part1.npy')
    pass