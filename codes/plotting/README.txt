All the plotting scripts require the data files (.npy) being stored in a local directory "./stats/".
The naming convention must follow env_agent_normalized_x.npy. (x in [0,9] with exception for Tabular x=42)
e.g. MiniGrid-LavaGapS6-v0_mlp_True_0.npy

plot_tabular.py [env] [spec]
Plot the [spec] of the Tabular agent on [env].
env - MiniGrid-Empty-6x6-v0 / MiniGrid-LavaGapS6-v0 / MiniGrid-DistShift1-v0
spec - 0 / 1 / 2
Note: 0 for coverage, 1 for returns, 2 for plot_entropy

plot_mlp.py [env] [spec]
Plot the spec of the MLP agent on env.
env - MiniGrid-Empty-6x6-v0 / MiniGrid-LavaGapS6-v0 / MiniGrid-DistShift1-v0
spec - 0 / 1 / 2
Note: 0 for coverage, 1 for returns, 2 for plot_entropy

plot_coverage.py [env]
Plot the coverage of the Tabular, MLP and EMLP agents on [env].
env - MiniGrid-Empty-6x6-v0 / MiniGrid-LavaGapS6-v0 / MiniGrid-DistShift1-v0

plot_returns.py [env]
Plot the returns of the Tabular, MLP and EMLP agents on [env].
env - MiniGrid-Empty-6x6-v0 / MiniGrid-LavaGapS6-v0 / MiniGrid-DistShift1-v0

plot_entropy.py [env]
Plot the entropy of the Tabular, MLP and EMLP agents on [env].
env - MiniGrid-Empty-6x6-v0 / MiniGrid-LavaGapS6-v0 / MiniGrid-DistShift1-v0

Example Usage:

To generate Figure 2(a)
plot_mlp.py MiniGrid-Empty-6x6-v0 0

To generate Figure 2(e)
plot_mlp.py MiniGrid-LavaGapS6-v0 2

To generate Figure 3(b)
plot_tabular.py MiniGrid-LavaGapS6-v0 0

To generate Figure 3(f)
plot_tabular.py MiniGrid-DistShift1-v0 2

To generate Figure 4(a)
plot_returns.py MiniGrid-Empty-6x6-v0

To generate Figure 4(e)
plot_coverage.py MiniGrid-LavaGapS6-v0

To generate Figure 4(i)
plot_entropy.py MiniGrid-DistShift1-v0
