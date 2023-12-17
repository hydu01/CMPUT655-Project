"""
Plot a spec of Tabular on a given environment.

command line arguments:
spec - coverage/returns/entropy
env - minigrid env
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

env = sys.argv[1]
i = int(sys.argv[2])
plot_typpe= {0: "coverage", 1: "returns", 2: "entropy"}
model = "mlp"
norm = True
name = "./stats/{}_{}_{}_{}.npy"
num_episode = None

######################################################################################

model = "tabular"
num_episode = 1000
norm = True

stats = np.load(name.format(env, model, norm, 42))

a = np.zeros((10, num_episode))
a = stats[i,:,:]

avg = np.mean(a, axis=0)
std = 1.96 * np.std(a, axis=0) / np.sqrt(a.shape[0])
plt.plot(avg, label="tabular + optim. init.")
plt.fill_between(np.arange(num_episode), avg-std, avg+std, alpha=0.1)

######################################################################################

model = "tabular"
num_episode = 1000
norm = False

stats = np.load(name.format(env, model, norm, 42))

a = np.zeros((10, num_episode))
a = stats[i,:,:]

avg = np.mean(a, axis=0)
std = 1.96 * np.std(a, axis=0) / np.sqrt(a.shape[0])
plt.plot(avg, label="tabular only")
plt.fill_between(np.arange(num_episode), avg-std, avg+std, alpha=0.1)

########################################################################################

plt.xlabel("episode")
plt.ylabel(plot_typpe[i])
plt.legend(loc="lower right")
plt.savefig(env+"_tabular_"+plot_typpe[i]+".pdf")
plt.show()
