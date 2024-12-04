import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters
OUT_DIR = "continuous"  # Output directory
EXP_ID_DEFAULT = "default"  # Default experiment ID
EXP_ID_MODIFIED = "changed"  # Modified experiment ID
functions = ["f01", "f02", "f06", "f08", "f10"]

# Helper to load minimum values (2nd column) from `.objective` files
def load_minimum_results(out_dir, exp_id, func_name):
    file_path = os.path.join(out_dir, f"{exp_id}.{func_name}_0.objective")  # Using the first repeat
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Objective file not found: {file_path}")
    # Extract the 2nd column (minimum values) from the file
    min_values = []
    with open(file_path, "r") as file:
        for line in file:
            try:
                min_values.append(float(line.split()[1]))  # Parse the 2nd column
            except (ValueError, IndexError):
                print(f"Skipping invalid line in {file_path}: {line.strip()}")
    return np.array(min_values)

# Load results for both default and modified experiments
default_results = {}
modified_results = {}

for func in functions:
    default_results[func] = load_minimum_results(OUT_DIR, EXP_ID_DEFAULT, func)
    modified_results[func] = load_minimum_results(OUT_DIR, EXP_ID_MODIFIED, func)

# Plotting comparison for each function
for func in functions:
    generations = np.arange(1, len(default_results[func]) + 1)

    plt.figure()
    plt.plot(generations, default_results[func], label="Default Operators")
    plt.plot(generations, modified_results[func], label="Modified Operators", linestyle="--")
    plt.yscale("log")  # Logarithmic scale for better visibility of differences
    plt.xlabel("Generation")
    plt.ylabel("Minimum Objective Value (log scale)")
    plt.title(f"Comparison of Default and Modified Operators: {func}")
    plt.legend()
    #plt.show()

    plt.close('all')
    #plt.close()
