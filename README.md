# 🔥 Hyper-Heuristic Simulated Annealing for Bin Packing Problem (BPP)

This project implements a **Hyper-Heuristic Simulated Annealing (SA)** algorithm to solve the **Bin Packing Problem (BPP)**. The algorithm dynamically selects and applies adaptive low-level heuristics to minimize the number of bins needed to pack a set of items, while ensuring that the total capacity of each bin is not exceeded.

## ✨ Features
- **Hyper-Heuristic** approach to optimize bin packing with adaptive learning.
- Multiple heuristic strategies:
  - 🔄 **Shift Heuristic**
  - ✂️ **Split Heuristic**
  - 📦 **Largest Bin Heuristic**
  - 🔀 **Shuffle Heuristic**
  - 🏆 **Best Packing Heuristic**
- Configurable parameters for temperature, iterations, and runs.
- Support for reading problems from a data file and saving solutions to an output file.

## 📚 Datasets
To test the algorithm, you can use the available Bin Packing Problem datasets from the [ESICUP Data Sets Repository](https://www.euro-online.org/websites/esicup/data-sets/).

### 📂 Input Dataset Format
The input dataset is structured as follows:
1. The first line contains the **number of test problems (P)**.
2. For each problem (p = 1,..., P), the following data is provided:
   - **Problem identifier** (a unique label for the problem).
   - Three values: **bin capacity**, **number of items (n)**, and the **best known number of bins** for the problem.
   - The next 'n' lines contain the **size of each item** to be packed.

## 📝 Research Basis
This work is based on the research presented in the PDF **"4or-sahh"**, which explores the use of Simulated Annealing within a hyper-heuristic framework for solving optimization problems like the Bin Packing Problem.

## 🚀 Usage
Run the program with the following command-line options:
```bash
./binpack -s <data_file> -o <output_file> -t <max_time>
