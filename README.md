# Optimistic Initialization of Non-linear Parameterized Value Functions

## Prerequisites
- Python >= 3.9.0,<3.11
- Poetry 1.5.0+
  
To locally build an environment, run a following command:
```
poetry install
```

To build a docker image and container, run following commands:
```
make build
make run
```
If you do not have `poetry.lock` file and would like to run a container,
run a `poetry install` locally to generate a lock file. 