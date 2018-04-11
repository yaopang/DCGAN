import numpy as np
from matplotlib import pyplot as plt
data = np.array([10,11,11,12,12,12,12,12,12,12,13,13,13,13,13,14,14,14,14,14,14,14,
                 15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,16,16,17,17,17,17,17,18,18,18,18,18,18,18,19,19])/20.
mean = np.mean(data)
median = np.median(data)
variance = np.var(data)

mean_string = 'Mean: ' + str(np.round(mean,3))
median_string = 'Median: ' + str(np.round(median,3))
std_string = 'Standard Dev: ' + str(np.round(np.sqrt(variance),3))

plt.hist(data)
plt.text(.6, 9, mean_string + '\n' + median_string + '\n' + std_string, horizontalalignment='center', verticalalignment='center',
         fontsize=14, bbox=dict(facecolor=(.7, .7, .7), alpha=0.3))
plt.xlabel("Score")
plt.ylabel("# of Respondents")
plt.show()
pass

