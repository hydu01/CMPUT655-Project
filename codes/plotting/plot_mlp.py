"""
Plot a spec of MLP on a given environment.

command line arguments:
spec - coverage/returns/entropy
env - minigrid env
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

env = sys.argv[1]
p = int(sys.argv[2])
l = 10
if env=="MiniGrid-DistShift1-v0":
    l = 7
plot_typpe= {0: "coverage", 1: "returns", 2: "entropy"}
model = "mlp"
norm = True
name = "./stats/{}_{}_{}_{}.npy"
num_episode = None

######################################################################################

model = "mlp"
num_episode = 1000
norm = True

runs = []
for i in range(l):
    runs.append(np.load(name.format(env, model, norm, i)))
num_episode = runs[0].shape[0]

a = np.zeros((l, num_episode))

for i in range(l):
    stats = runs[i]
    a[i,:] = stats[:,p]

avg = np.mean(a, axis=0)
std = 1.96 * np.std(a, axis=0) / np.sqrt(a.shape[0])
plt.plot(avg, label="mlp + optim. init.")
plt.fill_between(np.arange(num_episode), avg-std, avg+std, alpha=0.1)

######################################################################################

model = "mlp"
num_episode = 1000
norm = False

runs = []
for i in range(l):
    runs.append(np.load(name.format(env, model, norm, i)))
num_episode = runs[0].shape[0]

a = np.zeros((l, num_episode))

for i in range(l):
    stats = runs[i]
    a[i,:] = stats[:,p]

avg = np.mean(a, axis=0)
std = 1.96 * np.std(a, axis=0) / np.sqrt(a.shape[0])
plt.plot(avg, label="mlp only")
plt.fill_between(np.arange(num_episode), avg-std, avg+std, alpha=0.1)

########################################################################################

plt.xlabel("episode")
plt.ylabel(plot_typpe[p])
plt.legend(loc="lower right")
plt.savefig(env+"_mlp_"+plot_typpe[p]+".pdf")
plt.show()
