import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import spline


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def scale(x, out_range=(0, 100)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def read_data(filename):
    loss_data = {}
    for char in ['d', 'g']:
        actual_filename = filename.format(char)
        csv_data = pd.read_csv(actual_filename)
        loss_data.update({'{}_loss'.format(char): np.array(csv_data['Value'])})
        loss_data.update({'{}_time'.format(char): np.array(scale((csv_data['Wall time'] - csv_data['Wall time'][0])))})

    return loss_data


def plot_trio(data_list, names_list):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for i, data in enumerate(data_list):
        axes[i].plot(data['d_time'], smooth(data['d_loss'], 10))
        axes[i].plot(data['g_time'], smooth(data['g_loss'], 10))
        axes[i].set_title(names_list[i])
        axes[i].set_xlabel('epochs')
        axes[i].set_ylabel('loss')
        axes[i].legend(['d_loss', 'g_loss'])
        plot_max = max(np.int(np.max(data['d_loss'])), np.int(np.max(data['d_loss'])))
        d_string = 'final d_loss: ' + str(np.round(data['d_loss'][-1], 3))
        g_string = 'final g_loss: ' + str(np.round(data['g_loss'][-1], 3))
        axes[i].text(80, 4, d_string + '\n' + g_string, horizontalalignment='center', verticalalignment='center',
                     fontsize=14, bbox=dict(facecolor=(.7, .7, .7), alpha=0.3))

    plt.tight_layout()
    plt.show()


learning_filenames = ['documentation/{}_loss_lr0.02.csv',
                      'documentation/{}_loss_lr0.002.csv',
                      'documentation/{}_loss_lr0.0002.csv']

batch_filenames = ['documentation/{}_loss_normal.csv',
                   'documentation/{}_loss_batch_512.csv',
                   'documentation/{}_loss_batch_1024.csv']

layer_filenames = ['documentation/{}_loss_layer_3.csv',
                   'documentation/{}_loss_normal.csv',
                   'documentation/{}_loss_layer_5.csv']

opt_filenames = ['documentation/{}_loss_normal.csv',
                 'documentation/{}_loss_opt_adagrad.csv',
                 'documentation/{}_loss_opt_mom.csv']

learning_names = ['learning rate = 0.02', 'learning rate = 0.002', 'learning rate = 0.0002']
learning_data = []
for learning_filename in learning_filenames:
    learning_data.append(read_data(learning_filename))

batch_names = ['batch size = 256', 'batch size = 512', 'batch size = 1024']
batch_data = []
for batch_filename in batch_filenames:
    batch_data.append(read_data(batch_filename))

layer_names = ['num layers = 3', 'num layers = 4', 'num layers = 5']
layer_data = []
for layer_filename in layer_filenames:
    layer_data.append(read_data(layer_filename))

opt_names = ['optimizer = Adam', 'optimizer = AdaGrad', 'optimizer = Momentum']
opt_data = []
for opt_filename in opt_filenames:
    opt_data.append(read_data(opt_filename))

plot_trio(learning_data, learning_names)
plot_trio(batch_data, batch_names)
plot_trio(layer_data, layer_names)
plot_trio(opt_data, opt_names)