"""
Labeling script for the 271b project
Usage: python label_data.py path_image_dir/
where image_dir/ is the path from label_data.py to the directory of images you want to label.
If you choose to save progress, it will make a directory where label_data.py is named saved_progress/
If you want to change the name or the location the program will look, look above the "__main__" function.
"""

import numpy as np
import os
import cv2
import sys
import signal
import pickle


def get_images(directory):
    filenames = os.listdir(directory)
    filenames = sorted(filenames)
    known_image_types = ['png', 'jpg', 'JPG']
    valid = [False for _ in range(len(filenames))]
    for image_type in known_image_types:
         v = [image_type in filename for filename in filenames]
         valid = np.bitwise_or(valid, v)
    images = [cv2.cvtColor(cv2.imread(os.path.join(directory, filename)), cv2.COLOR_BGR2RGB) for i, filename in enumerate(filenames) if valid[i]]
    valid_filenames = [filename for i, filename in enumerate(filenames) if valid[i]]

    return valid_filenames, images


# def signal_handler(signal, frame):
#     if signal_handler.quit_status < 1:
#         print 'You pressed Ctrl+C!'
#         print 'If you would like to save your progress ' \
#               'select image window and press P, otherwise, pressing Ctrl+C again will exit.'
#         return
#     else:
#         sys.exit(0)


def get_save_directory():
    return get_save_directory.sd


def pause_handler(labels, save_directory='saved_progress/'):
    get_save_directory.sd = save_directory
    while True:
        progress_filename = raw_input("Progress Filename?\n>> ")
        if '.pickle' in progress_filename:
            break
        else:
            progress_filename += '.pickle'
            break

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(os.path.join(save_directory, progress_filename), 'wb') as f:
        pickle.dump(labels, f)

    print "Progress successfully saved to file [", save_directory + progress_filename, "]\n", len(labels.keys()), "images labeled!"


def save_labels(labels, save_directory='./'):

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(os.path.join(save_directory, 'labels.pickle'), 'wb') as f:
        pickle.dump(labels, f)

    print "Successfully saved labels to file [", save_directory + '/labels.pickle', "]\n", len(labels.keys()), "images labeled!"


def print_help():
    print "Use Left arrow key to mark image as FEMALE"
    print "Use Right arrow key to mark image as MALE"
    print "Use B key to go back an image"
    print "Use P key to pause progress\n"


def labeler(directory):
    print_help()
    # signal.signal(signal.SIGINT, signal_handler)
    num_done = 1
    filenames, images = get_images(directory)
    if os.path.exists(get_save_directory()):
        files = os.listdir(get_save_directory())
        if files:
            print "Saved Progress Files Found: ", files
            is_continue = raw_input("Continue from saved? (y/n) ")
            if is_continue == 'y' or is_continue == 'Y':
                continue_file = None
                while continue_file not in files and continue_file is None:
                    continue_file = raw_input("Which progress file? ")

                labels = load_progress(os.path.join(get_save_directory(), continue_file))
                done = [False for _ in range(len(filenames))]
                for key in labels.keys():
                    done = np.bitwise_or(done, [key == filename for filename in filenames])

                filenames = [filename for k, filename in enumerate(filenames) if not done[k]]
                images = [image for k, image in enumerate(images) if not done[k]]
                num_done = len(labels.keys())
                print "Successfully loaded progress file,", num_done, " labels loaded. Continuing..."

            elif is_continue == 'n' or is_continue == 'N':
                print "Starting fresh...\n"
                labels = {}
            else:
                print "Please try again, answering either (y or n)."
                return
        else:
            labels = {}
    i = 0
    pause_flag = 0
    label = -1
    while i < len(images):
        back_flag = 0
        while True:
            cv2.imshow("Image", cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)
            if key == 1113937:  # left arrow
                print i+num_done-1, ': Female'
                label = 0
                break
            elif key == 1113939:  # right arrow
                print i+num_done-1, ': Male'
                label = 1
                break
            elif key == 1048688:  # P
                print 'Pause at image ', i
                pause_flag = 1
                break
            elif key == 1048674:  # B
                i -= 1
                if i < 0:
                    i = 0
                print 'Back to image ', i+num_done-1
                labels.pop(filenames[i], None)
                back_flag = 1
                break
            elif key != -1:
                print_help()
        if pause_flag:
            pause_handler(labels)
            break
        if back_flag:
            continue
        labels.update({filenames[i]: label})
        i += 1

    save_labels(labels, directory)


def load_progress(filepath):
    with open(filepath, 'rb') as f:
        labels = pickle.load(f)
    return labels


# signal_handler.quit_status = 0
get_save_directory.sd = 'saved_progress/'
if __name__ == "__main__":
    images_directory = sys.argv[1]
    labeler(images_directory)
