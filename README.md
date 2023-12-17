# Optimistic Initialization of Non-linear Parameterized Value Functions

## Prerequisites
- Python >= 3.9.0,<3.11
- Poetry 1.5.0+

## Building a development environment
### Poetry (Mandatory)
To locally build an environment, run a following command:
```
poetry install
```
To run the code within the poetry environment, either:
1. Put a prefix of `poetry run` for each python command (e.g. `poetry run python3 FILE_NAME`), or
2. Build and activate the virtual environment by `poetry shell`

### Docker (Optional)
To build a docker image and container, run following commands:
```
make build
make run
```
If you do not have `poetry.lock` file and would like to run a container,
run a `poetry install` locally to generate a lock file. 

## Run the experiments
**Please run all the commands inside the `codes/` directory**
### SARSA
The experiments can be conducted by running a following command:
```
python3 sarsa.py /PATH/TO/CONFIG/FILE
```
Unlike in semi-grad SARSA (discussed below), the outputs are going to be produced under the `results/sarsa/` directory. As an example, one cna run SARSA with optimism on the EmptyEnv by running:
```
python3 sarsa.py ../configs/tabular_opt_emptyenv_config.json
```

### Semi-grad SARSA
The experiments can be conducted by running a following command:
```
python3 semi_grad_sarsa[_emlp].py CONFIG_FILE_NAME
```
Choose either `semi_grad_sarsa.py` or `semi_grad_sarsa_emlp.py` depending on the preferred architecture. The config files can be found in the `configs` directory. For instance, to run an experiment the optimistically initialized MLP on the EmtpyEnv, one can run:
```
python3 semi_grad_sarsa.py semi_grad_normalized_empty.json
```

### Notable options in the configs
Here is the list of important options on the occasion of running the custom experiments:
- `base_save_dir`: Directory to save all the output files.
- `model_type`: Set to "mlp" to use the MLP, and "emlp" to use the EMLP.
- `env_name`: Specify the **exact** name of [Minigrid environment](https://minigrid.farama.org/) that you would like to run.
- `normalize_reward`: Set to "true" to optimistically initialize, and "false" to randomly initialize.
- `n_seeds`: Number of different seeds to repeat the experiments.

## Plotting the results
The plotting can be done by placing all the output files (`.npy`) under the `codes/plotting/stats/` directory and then running the desired python file inside the `codes/plotting/` directory. For clear instructions, please refer `README.txt` placed in the `codes/plotting/` directory. 