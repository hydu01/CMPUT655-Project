import json
import matplotlib.pyplot as plt
import numpy as np
import glob

# Load JSON data from a file
if __name__ == '__main__':
    environments = ["DistShift1-v0", "Empty-Random-6x6-v0", "LavaGapS5-v0"]
    for env in environments:
        pattern = f'../results/true_values/true_values-MiniGrid-{env}-*'
        matching_files = glob.glob(pattern)
        data = []
        for file_path in matching_files:
            with open(file_path, 'r') as file:
                data.append(json.load(file))
        average_over_runs = {key: sum(d.get(key,0) for d in data) / len(data) for key in data[0]}

        # Extract coordinates and values
        coordinates = [eval(coord) for coord in average_over_runs.keys()]
        values = list(average_over_runs.values())

        # Get grid size
        max_x = max(coord[0] for coord in coordinates)
        max_y = max(coord[1] for coord in coordinates)
        if env == "LavaGapS5-v0": # handling this special case
            max_x = 3
            max_y = 3

        # Create a grid for values
        value_grid = np.zeros((max_y, max_x))

        # Fill the grid with values
        for coord, value in zip(coordinates, values):
            value_grid[coord[1] - 1, coord[0] - 1] = value

        # Create a heatmap for values
        plt.figure()
        plt.imshow(value_grid, cmap='viridis', interpolation='nearest')

        # Add text annotations for each cell
        for i in range(max_y):
            for j in range(max_x):
                plt.text(j, i, f'{value_grid[i, j]:.2f}', ha='center', va='center', color='orange')

        plt.colorbar()
        plt.title(f'State Values Overlay - {env}')
        plt.savefig(f"../results/true_value_overlay/true_values_{env}")
