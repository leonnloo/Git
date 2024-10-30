# Hyper-Heuristic Simulated Annealing for Bin Packing Problem (BPP)

This project implements a **Hyper-Heuristic Simulated Annealing (SA)** algorithm to solve the **Bin Packing Problem (BPP)**. The algorithm dynamically selects and applies adaptive low-level heuristics to minimize the number of bins needed to pack a set of items, while ensuring that the total capacity of each bin is not exceeded.

## Features
- **Hyper-Heuristic** approach to optimize bin packing with adaptive learning.
- Multiple heuristic strategies:
  - Shift Heuristic
  - Split Heuristic
  - Largest Bin Heuristic
  - Shuffle Heuristic
  - Best Packing Heuristic
- Configurable parameters for temperature, iterations, and runs.
- Support for reading problems from a data file and saving solutions to an output file.


## Usage
Run the program with the following command-line options:
```bash
./binpack -s <data_file> -o <output_file> -t <max_time>
