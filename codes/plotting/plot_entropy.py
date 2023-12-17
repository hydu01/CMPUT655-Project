"""
Plot the entropy of Tabular, MLP, and EMLP
on the environment given by the command line argument.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

env = sys.argv[1]
model = "mlp"
norm = True
name = "./stats/{}_{}_{}_{}.npy"
num_episode = None

######################################################################################

model = "tabular"
num_episode = 1000

stats = np.load(name.format(env, model, norm, 42))

returns_linear = np.zeros((10, num_episode))
coverage_linear = np.zeros((10, num_episode))
entropy_linear = np.zeros((10, num_episode))

returns_linear = stats[1,:,:]
coverage_linear = stats[0,:,:]
entropy_linear = stats[2,:,:]

avg_linear = np.mean(entropy_linear, axis=0)
std_linear = 1.96 * np.std(entropy_linear, axis=0) / np.sqrt(entropy_linear.shape[0])
plt.plot(avg_linear, label="tabular")
plt.fill_between(np.arange(num_episode), avg_linear-std_linear, avg_linear+std_linear, alpha=0.1)

######################################################################################

model = "emlp"
num_episode = None

runs = []
for i in range(10):
    runs.append(np.load(name.format(env, model, norm, i)))
num_episode = runs[0].shape[0]

returns_emlp = np.zeros((10, num_episode))
coverage_emlp = np.zeros((10, num_episode))
entropy_emlp = np.zeros((10, num_episode))

for i in range(10):
    stats = runs[i]
    returns_emlp[i,:] = stats[:,1]
    coverage_emlp[i,:] = stats[:,0]
    entropy_emlp[i,:] = stats[:,2]

avg_emlp = np.mean(entropy_emlp, axis=0)
std_emlp = 1.96 * np.std(entropy_emlp, axis=0) / np.sqrt(entropy_emlp.shape[0])
plt.plot(avg_emlp, label="emlp")
plt.fill_between(np.arange(num_episode), avg_emlp-std_emlp, avg_emlp+std_emlp, alpha=0.1)

########################################################################################

model = "mlp"
num_episode = None

runs = []
for i in range(10):
    runs.append(np.load(name.format(env, model, norm, i)))
num_episode = runs[0].shape[0]

returns_mlp = np.zeros((10, num_episode))
coverage_mlp = np.zeros((10, num_episode))
entropy_mlp = np.zeros((10, num_episode))

for i in range(10):
    stats = runs[i]
    returns_mlp[i,:] = stats[:,1]
    coverage_mlp[i,:] = stats[:,0]
    entropy_mlp[i,:] = stats[:,2]

avg_mlp = np.mean(entropy_mlp, axis=0)
std_mlp = 1.96 * np.std(entropy_mlp, axis=0) / np.sqrt(entropy_mlp.shape[0])
plt.plot(avg_mlp, label="mlp")
plt.fill_between(np.arange(num_episode), avg_mlp-std_mlp, avg_mlp+std_mlp, alpha=0.1)

########################################################################################

plt.xlabel("episode")
plt.ylabel("entropy")
plt.legend(loc="lower right")
plt.savefig(env+"_"+"entropy.pdf")
plt.show()
