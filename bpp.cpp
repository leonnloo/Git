/**
 * @file 20410097.cpp
 * @author Leon (https://github.com/leonnloo)
 * @brief 
 * @version 0.1
 * @date 2024-10-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <climits>
#include <map>
#include <chrono>

// Global parameters
char data_file[500]={"somefile"}, out_file[500]={}, solution_file[500]={};  // max 500 problem instances per run
int NUM_OF_RUNS = 1;                                    // Number of run per instance
int MAX_TIME = 30;                                      // Max amount of time permitted (in sec)
int num_of_problems;                                    // Number of problem instance
int NUM_OF_BINS;                                        // Number of bins for bin packing
float random_check;

// Parameters for simulated annealing
int number_of_heuristics = 5;                           // Number of heuristics
float temperature_start = 1;                            // Starting temperature
float temperature_end = 0.005;                          // Ending temperature
float temperature = temperature_start;
float temperature_improvement = temperature_start;
int temperature_iteration = number_of_heuristics;       // Number of iterations to cool temperature
int SA_MAX_iteration = 110000;                          // Number of iteration per run
int counter_accepted_heuristics = 0;                    // Total number of heuristics called
int iter = 0;                                           // Current iteration on SA
float ratio_start = 0.1;
float ratio_end = 0.005;
bool reheat = false;                                    // Flag indicating to reheat temperature
bool worse = false;                                     // Flag indicating accept worse solution
float beta = (temperature_start - temperature_end) * temperature_iteration / (static_cast<float>(SA_MAX_iteration) * temperature_start * temperature_end);
float wmin = std::min(100 * static_cast<float>(number_of_heuristics) / static_cast<float>(SA_MAX_iteration), static_cast<float>(0.1));
int global_best;                                        // Best known objective for current instance

// Dynamically define the length of the learning period (LP)
const int LP = std::max(SA_MAX_iteration/500, number_of_heuristics); 

// Type of LLH available
enum LowLevelHeuristicType { SHIFT, SPLIT, LARGEST_BIN, SHUFFLE, BEST_PACKING };

// Structure for item
struct Item {
    int size;                           // Size of the item
    int index;                          // Index of the item

    // Compare items
    bool operator==(const Item& other) const {
        return size == other.size && index == other.index;
    }
};

// Structure for bins
struct bin {
    std::vector<Item> items;            // Items in the bin
    int left_bin_capacity;              // Residual Capacity of bin
};

// Structure for problem
struct Problem {
    int bin_capacity;                   // Bin capacity;
    int n;                              // Number of items
    int best_obj;                       // Best known objective
    std::string problem_identifier;     // Problem name
    std::vector<Item> items;            // Items of the problem
};

struct LowLevelHeuristic {
    LowLevelHeuristicType type;         // Low Level Heuristic type
    int total_counter;                  // Total number of times the heuristic is called
    int accepted_counter;               // Number of times heuristic generate better solutions
    int new_counter;                    // Number of times heuristic generate new solutions
    float weight;                       // Weight of the heuristic for probability selection
    float weight_min;                   // Minimum weight to ensure heuristic can still be selected
};

// Structure for solution
struct Solution {
    std::vector<bin> bins;              // Bins of the solution
    int objective;                      // Number of bins
    int bin_capacity;                   // Bin capacity
    LowLevelHeuristicType heuristic;    // Current Low Level Heuristic type
    Problem* prob;                      // Maintain a shallow copy of the problem data
};

// global best solution
Solution best_sln;

// Global random number generator
std::mt19937 g;  // Declare the random generator

// Function to initialize the random generator with a seed
void init_random_generator(int seed) {
    g.seed(seed);
}

// ================================================= FUNCTIONS ======================================================
// Random float generator from 0 to 1
float rand_01() {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    return dist(g);
}

// Load problems from a data file
std::vector<Problem*> load_problems(const char* data_file) {
    std::ifstream infile(data_file);
    if (!infile) {
        std::cerr << "Data file " << data_file << " does not exist. Please check!\n";
        exit(2);
    }

    infile >> num_of_problems; // Read number of problem instance in the file

    // Initialize number of problem instance
    std::vector<Problem*> problems(num_of_problems);

    for (int k = 0; k < num_of_problems; k++) {
        std::string problem_identifier;

        // read problem identifier
        infile >> problem_identifier;

        // read bin capacity, number of items in instance, best possible bin number
        int bin_capacity, num_of_items, best_bin_sol;
        infile >> bin_capacity >> num_of_items >> best_bin_sol;
        
        // create a new problem instance
        Problem* problem = new Problem;
        problem->problem_identifier = problem_identifier;
        problem->bin_capacity = bin_capacity;
        problem->n = num_of_items;
        problem->best_obj = best_bin_sol;

        // read item sizes
        for (int j = 0; j < num_of_items; j++) {
            int item_size;
            infile >> item_size;
            problem->items.push_back({item_size, j});
        }

        // push current problem instance into a list of problems
        problems[k] = problem;
    }

    infile.close();

    // return all problems read
    return problems;
}

// Function to output the best solution to a file
void output_solution(const Solution& current_solution, float abs_gap ,const char* out_file) {
    std::ofstream ofs(out_file, std::ios::app); // Append solution data
    ofs << current_solution.prob->problem_identifier << "\n";
    ofs << "obj= " << current_solution.objective << " " << current_solution.objective - global_best << std::endl;
    for (int i = 0; i < current_solution.bins.size(); i++) {
        for (int j = 0; j < current_solution.bins[i].items.size(); j++) {
            ofs << current_solution.bins[i].items[j].index << " ";
        }
        ofs << std::endl;
    }
    ofs.close();
}

// Dellocate memory
void free_problems(std::vector<Problem*>& problems) {
    for (Problem* prob : problems) {
        delete prob;
    }
    problems.clear();
}

// ============================================= LOW LEVEL HEURISTICS ===================================================================
// Initialize the low-level heuristic instances
LowLevelHeuristic shift_heuristic_instance;
LowLevelHeuristic split_heuristic_instance;
LowLevelHeuristic largest_bin_heuristic_instance;
LowLevelHeuristic shuffle_heuristic_instance;
LowLevelHeuristic best_packing_heuristic_instance;
std::vector<LowLevelHeuristic> low_level_heuristics;

// LLH map for accessing index of LLH within the low_level_heuristics vector
std::map<LowLevelHeuristicType, int> heuristicIndexMap = {
    {SHIFT, 0},
    {SPLIT, 1},
    {LARGEST_BIN, 2},
    {SHUFFLE, 3},
    {BEST_PACKING, 4}
};

// compare item index
bool compareItems(const Item& a, const Item& b) {
    return a.index == b.index; 
}

// Function to compare two solutions for equality, where each solution consists of multiple bins
bool compare_same_solution(const Solution& new_solution, const Solution& candidate_solution) {
    // Return false immediately if the number of bins in both solutions is not the same
    if (new_solution.bins.size() != candidate_solution.bins.size())
        return false;

    // Iterate through each bin in the new solution
    for (const auto& new_bin : new_solution.bins) {
        bool bin_found = false; // Flag to check if a matching bin is found in the candidate solution

        // Compare the current bin of the new solution with each bin in the candidate solution
        for (const auto& current_bin : candidate_solution.bins) {
            // Return false if the number of items in the bins are not the same
            if (new_bin.items.size() != current_bin.items.size()) {
                return false;
            }
            bool all_items_match = true; // Flag to check if all items in the current bin match

            // Check each item in the new bin against items in the current bin of the candidate solution
            for (const auto& item : new_bin.items) {
                // Use std::find_if with a lambda function to check if the item matches any item in the current bin
                if (std::find_if(current_bin.items.begin(), current_bin.items.end(),
                                 [&item](const Item& other) { return compareItems(item, other); }) == current_bin.items.end()) {
                    all_items_match = false; // Set flag to false if an item does not match
                    break;
                }
            }
            // If all items match, set bin_found to true and break the inner loop
            if (all_items_match) {
                bin_found = true;
                break;
            }
        }
        // If no matching bin is found after checking all candidate bins, return false
        if (!bin_found) {
            return false;
        }
    }
    // Return true if all bins from the new solution match with bins in the candidate solution
    return true;
}


// Function to apply a shift heuristic to redistribute items between bins in a solution to optimize bin usage
Solution shift_heuristic(Solution& solution) {
    // Identify the bin with the largest remaining capacity
    int largest_bin_index = 0;
    int largest_residual = 0;
    for (int i = 0; i < solution.bins.size(); i++) {
        if (solution.bins[i].left_bin_capacity > largest_residual) {
            largest_residual = solution.bins[i].left_bin_capacity;
            largest_bin_index = i;
        }
    }

    // Reference to the bin with the largest residual capacity
    auto& current_bin = solution.bins[largest_bin_index];

    // Iterate over all bins to attempt item redistribution
    for (int i = 0; i < solution.bins.size(); i++) {
        if (i != largest_bin_index) { // Ensure not to target the same bin
            bin& original_bin = solution.bins[largest_bin_index];
            bin& target_bin = solution.bins[i];

            // Sort items in the original bin by size in descending order to prioritize larger items for shifting
            std::sort(original_bin.items.begin(), original_bin.items.end(), [](const Item& a, const Item& b) {
                return a.size > b.size;
            });

            // Attempt to move items from the largest bin to the target bin
            for (auto it = original_bin.items.begin(); it != original_bin.items.end();) {
                if (target_bin.left_bin_capacity >= it->size) {
                    target_bin.items.push_back(*it); // Add item to target bin
                    target_bin.left_bin_capacity -= it->size; // Adjust the capacity of the target bin
                    it = original_bin.items.erase(it); // Erase the item from the original bin and update iterator
                    original_bin.left_bin_capacity += it->size; // Reclaim the space in the original bin
                } else {
                    ++it; // Move to the next item if the current item cannot be moved
                }
            }

            // If the original bin is empty after moving items, stop the process
            if (original_bin.items.empty()){
                break;
            }
        }
        // Stop if there are no items left in the largest bin to consider for shifting
        if (solution.bins[largest_bin_index].items.empty()){
            break;
        }
    }

    // Remove any bins that have become empty after the item shifts
    solution.bins.erase(std::remove_if(solution.bins.begin(), solution.bins.end(),
                                       [](const bin& b) { return b.items.empty(); }),
                        solution.bins.end());

    // Update the solution metrics based on the shift heuristic
    solution.heuristic = SHIFT;
    solution.objective = solution.bins.size();
    return solution; // Return the updated solution
}


// Function to apply a split heuristic to redistribute items between bins more efficiently in a solution
Solution split_heuristic(Solution& solution) {
    size_t total_items = 0;
    // Calculate the total number of items across all bins
    for (const auto& bin : solution.bins) {
        total_items += bin.items.size();
    }
    size_t average_items_per_bin = total_items / solution.bins.size();              // Determine the average number of items per bin

    // Create and shuffle a vector of bin indices for random access
    std::vector<size_t> indices(solution.bins.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    // Iterate through shuffled indices to consider each bin for splitting
    for (auto index : indices) {
        auto& current_bin = solution.bins[index];

        // Proceed with splitting if the current bin has more items than the average
        if (current_bin.items.size() > average_items_per_bin) {
            // Initialize a new bin
            bin new_bin;
            new_bin.left_bin_capacity = solution.bin_capacity;                      // Set the new bin's capacity to the solution's default
            
            // Shuffle items in the current bin before moving half to the new bin
            std::shuffle(current_bin.items.begin(), current_bin.items.end(), g);

            size_t num_items_to_move = current_bin.items.size() / 2;                // Determine the number of items to move
            // Move half of the items to the new bin
            for (size_t k = 0; k < num_items_to_move; ++k) {
                new_bin.items.push_back(current_bin.items.back());
                new_bin.left_bin_capacity -= current_bin.items.back().size;         // Update the new bin's capacity
                current_bin.left_bin_capacity += current_bin.items.back().size;     // Update the old bin's capacity
                current_bin.items.pop_back();                                       // Remove the item from the old bin
            }
            solution.bins.push_back(new_bin);                                       // Add the new bin to the solution
            break;
        }
    }

    // Recalculate the objective value based on the new number of bins
    solution.heuristic = SPLIT;
    solution.objective = solution.bins.size();
    return solution; // Return the updated solution
}


// Function to perform an item exchange between bins in a solution, targeting the bin with the largest item and residual capacity
Solution exchange_largest_bin_largest_item(Solution& solution) {
    int largestBinIndex = -1;
    Item largestItem{0, -1}; // Struct to hold the largest item found

    // Identify the bin with the largest residual capacity
    int maxResidualCapacity = -1;
    for (size_t i = 0; i < solution.bins.size(); ++i) {
        if (solution.bins[i].left_bin_capacity > maxResidualCapacity) {
            maxResidualCapacity = solution.bins[i].left_bin_capacity;
            largestBinIndex = i;
        }
    }

    // If a bin with residual capacity was found
    if (largestBinIndex != -1) {
        auto& binFrom = solution.bins[largestBinIndex];
        // Find the largest item in the selected bin
        for (auto& item : binFrom.items) {
            if (item.size > largestItem.size) {
                largestItem = item;
            }
        }

        // Shuffle bins indices to randomize the selection of target bin
        std::vector<size_t> bin_indices(solution.bins.size());
        std::iota(bin_indices.begin(), bin_indices.end(), 0);
        std::shuffle(bin_indices.begin(), bin_indices.end(), g);

        // Attempt to find a suitable bin to make an exchange
        for (auto j : bin_indices) {
            if (j != largestBinIndex) {                                 // Ensure not to select the same bin
                auto& binTo = solution.bins[j];
                int spaceNeeded = largestItem.size;
                int smallerSize = 0;
                std::vector<Item> tempItems;

                // Collect smaller items from target bin to potentially exchange
                for (auto& item : binTo.items) {
                    if (item.size <= spaceNeeded) {
                        tempItems.push_back(item);
                        spaceNeeded -= item.size;
                        smallerSize += item.size;
                        if (spaceNeeded <= 0) break;                    // Found enough items to free up space
                    }
                }

                // Check if there is sufficient capacity to perform the exchange
                if (solution.bins[j].left_bin_capacity + smallerSize >= largestItem.size){

                    // Exchange the largest item with the smaller items
                    binFrom.items.erase(std::remove_if(binFrom.items.begin(), binFrom.items.end(),
                                                       [&](const Item& it) { return it.index == largestItem.index; }),
                                                       binFrom.items.end());
                    binTo.items.push_back(largestItem);
                    binTo.left_bin_capacity -= largestItem.size;
                    binFrom.left_bin_capacity += largestItem.size;

                    // Move the collected smaller items to the original bin
                    for (auto& item : tempItems) {
                        binTo.items.erase(std::remove_if(binTo.items.begin(), binTo.items.end(),
                                                         [&](const Item& it) { return it.index == item.index; }),
                                                         binTo.items.end());
                        binFrom.items.push_back(item);
                        binTo.left_bin_capacity += item.size;
                        binFrom.left_bin_capacity -= item.size;
                    }

                    break; // Exit the loop after successful exchange
                }
            }
        }
    }

    // Remove any empty bins as a result of the exchanges
    solution.bins.erase(
        std::remove_if(solution.bins.begin(), solution.bins.end(),
            [](const bin& b) { return b.items.empty(); }),
        solution.bins.end()
    );

    // Update the solution's objective
    solution.heuristic = LARGEST_BIN;
    solution.objective = solution.bins.size();
    return solution; // Return the modified solution
}



// Function to find the best combination of items that maximally fills one bin.
std::vector<Item> find_best_combination_one_bin(const std::vector<Item>& items, int bin_capacity) {
    std::vector<Item> best_combination;
    int best_combination_leftover_cap = INT_MAX;                                    // Initialize to maximum to find the minimum leftover capacity

    // Iterate through all possible combinations using bit masking
    for (size_t i = 0; i < (1ULL << items.size()); ++i) {                           // Make sure the shift is within the bounds of size_t
        std::vector<Item> current_combination;
        int current_combination_cap = bin_capacity;

        for (size_t j = 0; j < items.size(); ++j) {
            if (i & (1ULL << j)) {                                                  // Check if the j-th item is included in the i-th combination
                if (current_combination_cap >= items[j].size) {                     // Only add if it fits
                    current_combination.push_back(items[j]);
                    current_combination_cap -= items[j].size;
                }
            }
        }

        // Check until a better combination is found
        if (current_combination_cap >= 0 && current_combination_cap < best_combination_leftover_cap) {
            best_combination = current_combination;                                 // Update the best combination
            best_combination_leftover_cap = current_combination_cap;
        }
    }
    // Return the list of items that form the best combination to maximally fill one bin.
    return best_combination;
}

// Function to exchange and reshuffle items between two randomly selected bins based on their residual capacities
Solution exchange_random_bin_reshuffle(Solution& solution) {
    if (solution.bins.empty()) return solution;                                     // Return unchanged solution if no bins exist

    // Create a weighted distribution for bin selection based on residual capacities
    std::vector<std::pair<int, int>> binCapacities;                                 // Pair of bin index and its residual capacity
    std::vector<int> weights;                                                       // Weights for the distribution, based on residual capacity

    // Populate the vectors for the distribution
    for (size_t i = 0; i < solution.bins.size(); ++i) {
        if (solution.bins[i].left_bin_capacity > 0) {
            binCapacities.push_back({i, solution.bins[i].left_bin_capacity});
            weights.push_back(solution.bins[i].left_bin_capacity);
        }
    }
    if (binCapacities.size() < 2) return solution;                                  // If less than two bins have space, no reshuffling is possible

    std::discrete_distribution<> dist(weights.begin(), weights.end());              // Distribution based on the weights
    int bin1 = binCapacities[dist(g)].first;                                        // Select first bin
    int bin2 = bin1;
    while (bin2 == bin1) {
        bin2 = binCapacities[dist(g)].first;                                        // Select a different second bin
    }

    // Combine items from both bins into one list
    std::vector<Item> combinedItems(solution.bins[bin1].items);
    combinedItems.insert(combinedItems.end(), solution.bins[bin2].items.begin(), solution.bins[bin2].items.end());

    // Clear the selected bins to redistribute items by resetting items and their capacity
    solution.bins[bin1].items.clear();
    solution.bins[bin2].items.clear();
    solution.bins[bin1].left_bin_capacity = solution.bin_capacity;
    solution.bins[bin2].left_bin_capacity = solution.bin_capacity;

    // Redistribute items optimally for bin1
    auto bestCombinationBin1 = find_best_combination_one_bin(combinedItems, solution.bin_capacity);
    for (const auto& item : bestCombinationBin1) {
        solution.bins[bin1].items.push_back(item);
        solution.bins[bin1].left_bin_capacity -= item.size;
        combinedItems.erase(std::remove(combinedItems.begin(), combinedItems.end(), item), combinedItems.end());
    }

    // Place the remaining items in bin2
    for (const auto& item : combinedItems) {
        if (solution.bins[bin2].left_bin_capacity >= item.size) {
            solution.bins[bin2].items.push_back(item);
            solution.bins[bin2].left_bin_capacity -= item.size;
        }
    }

    // Remove any bins that have become empty as a result of this process
    solution.bins.erase(std::remove_if(solution.bins.begin(), solution.bins.end(),
                                       [](const bin& b) { return b.items.empty(); }),
                        solution.bins.end());

    // Recalculate the objective to reflect the current number of bins
    solution.heuristic = SHUFFLE;
    solution.objective = solution.bins.size();
    return solution; // Return the modified solution
}

// Function to collect a feasible combination of items from multiple bins within a given time limit,
// accounting for a predefined large item already included in the combination
std::vector<Item> time_bounded_item_collection(const std::vector<Item>& items, int initial_bin_capacity, Item& largestItem, Solution& solution) {
    auto start_time = std::chrono::steady_clock::now(); // Start timing
    double time_limit = 0.02;                                               // Set the time limit for the operation
    std::vector<Item> best_combination;                                     // This will hold the best combination of items found within the time limit

    for (int i = 0; i < solution.bins.size(); i++) {                        // Iterate over each bin in the solution
        int bin_capacity = initial_bin_capacity - largestItem.size;         // Adjust the bin's capacity to account for the largest item
        bin& bin = solution.bins[i];
        best_combination.clear();                                           // Clear previous combinations from other bins
        for (int j = 0; j < bin.items.size(); j++) {                        // Iterate over each item in the bin
            const Item& item = bin.items[j];
            // Check if the current operation time has exceeded the time limit
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() > time_limit) {
                break;                                                      // Exit the loop if the time limit is exceeded
            }

            // Add the item to the combination if it fits in the remaining capacity
            if (bin_capacity - item.size >= 0) {
                best_combination.push_back(item);
                bin_capacity -= item.size;                                  // Decrease the remaining capacity by the item's size
            }
        }
    }
    best_combination.push_back(largestItem);                                // Ensure the largest item is included in the final combination
    return best_combination;                                                // Return the collection of items
}

// Function to optimally redistribute items within a solution's bins to maximize bin utilization
Solution best_packing(Solution& solution) {
    // Compute total residual capacity across all bins
    float totalResidual = std::accumulate(solution.bins.begin(), solution.bins.end(), 0.0f,
                                          [](float sum, const bin& b) { return sum + b.left_bin_capacity; });

    // Create a weighted distribution for bin selection based on residual capacities
    std::vector<float> weights;
    for (const auto& bin : solution.bins) {
        weights.push_back(static_cast<float>(bin.left_bin_capacity) / totalResidual);   // Normalize weights
    }
    std::discrete_distribution<> dist(weights.begin(), weights.end());                  // Distribution based on the weights

    // Select a bin probabilistically based on the distribution
    int selectedBinIndex = dist(g);
    bin& selectedBin = solution.bins[selectedBinIndex];

    // Find the largest item in the selected bin
    auto largestItemIt = std::max_element(selectedBin.items.begin(), selectedBin.items.end(),
                                          [](const Item& a, const Item& b) { return a.size < b.size; });
    Item largestItem = *largestItemIt;
    selectedBin.items.erase(largestItemIt);                                             // Remove the largest item from its bin

    // Map to hold the residual capacity of the bin each item is currently in
    std::map<int, int> itemToBinResidualCapacity;
    for (const auto& bin : solution.bins) {
        for (const Item& item : bin.items) {
            itemToBinResidualCapacity[item.index] = bin.left_bin_capacity;
        }
    }

    // Collect all items from all bins, starting with the largest item
    std::vector<Item> allItems = {largestItem};
    for (const auto& bin : solution.bins) {
        allItems.insert(allItems.end(), bin.items.begin(), bin.items.end());
    }

    // Sort items by their bins' residual capacity to prioritize those with more space
    std::sort(allItems.begin(), allItems.end(), [&itemToBinResidualCapacity](const Item& a, const Item& b) {
        return itemToBinResidualCapacity[a.index] > itemToBinResidualCapacity[b.index];
    });

    // Apply a time-bounded heuristic to find an optimal packing for the selected items
    auto packedItems = time_bounded_item_collection(allItems, solution.bin_capacity, largestItem, solution);

    // Remove all packed items from their original bins
    for (auto& bin : solution.bins) {
        bin.items.erase(std::remove_if(bin.items.begin(), bin.items.end(), 
            [&packedItems](const Item& item) {
                return std::find_if(packedItems.begin(), packedItems.end(), 
                                    [&item](const Item& packedItem) { return item.index == packedItem.index; }) != packedItems.end();
            }), bin.items.end());
    }

    // Create a new bin for the packed items
    bin newBin;
    newBin.left_bin_capacity = solution.bin_capacity;
    for (const Item& item : packedItems) {
        newBin.items.push_back(item);
        newBin.left_bin_capacity -= item.size;
    }
    solution.bins.push_back(newBin);

    // Remove any empty bins resulting from the repacking
    solution.bins.erase(
        std::remove_if(solution.bins.begin(), solution.bins.end(),
            [](const bin& b) { return b.items.empty(); }),
        solution.bins.end()
    );

    // Update the solution's metrics to reflect the changes
    solution.heuristic = BEST_PACKING;
    solution.objective = solution.bins.size();
    return solution; // Return the optimized solution
}

// Function to select and apply a low-level heuristic to a solution based on a weighted probability
Solution apply_weighted_low_level_heuristic(Solution solution) {
    float total_weight = 0.0;
    // Sum the weights of all low-level heuristics
    for (const auto& heuristic : low_level_heuristics) {
        total_weight += low_level_heuristics[heuristicIndexMap[heuristic.type]].weight;
    }

    std::uniform_real_distribution<> dis(0.0, total_weight); // Distribution to select a point within the total weight
    float randomPoint = dis(g); // Generate a random point

    // Determine which heuristic to apply based on the random point
    float cumulativeWeight = 0.0;
    for (const auto& heuristic : low_level_heuristics) {
        auto& selectedHeuristic = low_level_heuristics[heuristicIndexMap[heuristic.type]];
        cumulativeWeight += selectedHeuristic.weight;
        if (randomPoint <= cumulativeWeight) {
            switch (selectedHeuristic.type) {
                case SHIFT: 
                    return shift_heuristic(solution);
                case SPLIT: 
                    return split_heuristic(solution);
                case LARGEST_BIN: 
                    return exchange_largest_bin_largest_item(solution);
                case SHUFFLE: 
                    return exchange_random_bin_reshuffle(solution);
                case BEST_PACKING: 
                    return best_packing(solution);
                default:
                    break; // No default action specified
            }
        }
    }
    return solution; // Return the solution unmodified if no heuristic matches (should not occur due to the total weight coverage)
}

// Function to insert an item into the bin with the minimum residual space (slack) after insertion
Solution minimum_bin_slack(Solution& solution, const Item& item) {
    int bestBinIndex = -1;
    int minSlack = INT_MAX;                                                 // Initialize the minimum slack to the maximum possible value

    // Iterate through each bin to find the one that can fit the item with the least slack
    for (size_t i = 0; i < solution.bins.size(); ++i) {
        int currentSlack = solution.bins[i].left_bin_capacity - item.size;  // Calculate slack for the current bin

        // Check if the current bin can fit the item and has less slack than the previously found bins
        if (currentSlack >= 0 && currentSlack < minSlack) {
            bestBinIndex = i;                                               // Update the index of the best bin
            minSlack = currentSlack;                                        // Update the minimum slack
        }
    }

    // If no suitable bin is found, create a new bin for the item
    if (bestBinIndex == -1) {
        bin newBin;
        newBin.left_bin_capacity = solution.bin_capacity - item.size;       // Adjust the capacity for the new bin
        newBin.items.push_back(item);                                       // Add the item to the new bin
        solution.bins.push_back(newBin);                                    // Add the new bin to the solution
    } else {
        solution.bins[bestBinIndex].items.push_back(item);                  // Add the item to the bin with the minimum slack
        solution.bins[bestBinIndex].left_bin_capacity -= item.size;         // Decrease the bin's capacity by the size of the item
    }

    return solution; // Return the updated solution
}

// Initial solution generator using MBS
Solution create_initial_solution_with_mbs(Problem* problem) {
    Solution solution;
    solution.prob = problem;
    solution.bin_capacity = problem->bin_capacity;
    solution.bins.clear();

    // Sort items by size to potentially enhance packing
    std::sort(problem->items.begin(), problem->items.end(), [](const Item& a, const Item& b) {
        return a.size > b.size;  // Sort by size descending
    });

    // Insert each item into the bin using minimum bin slack until all items are filled in a bin
    for (const Item& item : problem->items) {
        solution = minimum_bin_slack(solution, item);
    }

    // Calculate the initial objective value
    solution.objective = solution.bins.size();
    return solution;
}

// Initialize initial solution
Solution init_solution(Problem* problem) {
    return create_initial_solution_with_mbs(problem);
}

// ======================================= LEARNING PROCEDURE ======================================================
// Function to update heuristic learning based on the performance of the applied heuristics
void learn(Solution& candidate_solution, Solution& current_solution, int delta) {
    // Retrieve the heuristic used in the candidate solution
    LowLevelHeuristic& current_heuristic = low_level_heuristics[heuristicIndexMap[candidate_solution.heuristic]];

    // Increment the total usage counter for the current heuristic
    current_heuristic.total_counter++;

    // Check if the candidate solution is different from the current best solution
    if (!compare_same_solution(candidate_solution, current_solution)) {
        current_heuristic.new_counter++;                                    // Increment the counter for generating new solutions
    }

    // If the new solution is better (delta < 0), update acceptance counters and temperature
    if (delta < 0) {
        current_heuristic.accepted_counter++;                               // Increment the counter for accepted heuristics
        counter_accepted_heuristics++;                                      // Increment the global counter for accepted heuristics
        temperature_improvement = temperature;                              // Update the temperature improvement
        reheat = false;                                                     // Reset the reheat flag
    }

    // If the current solution is worse but accepted, update the counters
    if (worse) {
        current_heuristic.accepted_counter++;                               // Increment the counter for accepted heuristics
        counter_accepted_heuristics++;                                      // Increment the global counter for accepted heuristics
        worse = false;                                                      // Reset the worse flag
    }

    // Periodically update learning parameters and heuristics' weights
    if (iter % LP == 0) {
        // Adjust learning parameters based on the performance ratio
        if (static_cast<float>(counter_accepted_heuristics) / static_cast<float>(LP) < ratio_end) {
            reheat = true;                                                                              // Enable reheating
            temperature_improvement = temperature_improvement / (1 - (beta * temperature_improvement)); // Adjust the temperature
            temperature = temperature_improvement;                                                      // Update the global temperature
            current_solution = best_sln;                                                                // Reset the current solution to the best solution

            // Update weights based on the new solution generation rate
            for (auto& heuristic : low_level_heuristics) {
                if (heuristic.total_counter == 0) {
                    heuristic.weight = heuristic.weight_min;                                            // Reset weight to minimum if no uses
                } else {
                    heuristic.weight = std::max(heuristic.weight_min, static_cast<float>(heuristic.new_counter) / static_cast<float>(heuristic.total_counter)); // Update weight based on performance
                }
                heuristic.new_counter = 0;
                heuristic.accepted_counter = 0;
                heuristic.total_counter = 0;
            }
        } else {
            // Update weights based on the acceptance rate
            for (LowLevelHeuristic& heuristic : low_level_heuristics) {
                if (heuristic.total_counter == 0) {
                    heuristic.weight = heuristic.weight_min;                                            // Reset weight to minimum if no uses
                } else {
                    heuristic.weight = std::max(heuristic.weight_min, static_cast<float>(heuristic.accepted_counter) / static_cast<float>(heuristic.total_counter)); // Update weight based on acceptance rate
                }
                heuristic.new_counter = 0;
                heuristic.accepted_counter = 0;
                heuristic.total_counter = 0;
            }
        }
        counter_accepted_heuristics = 0; // Reset the global counter for accepted heuristics
    }
}

// ======================================= SIMULATED ANNEALING ======================================================
// Reset all variables for new problem instance
void reset(){
    temperature = temperature_start;
    temperature_improvement = temperature_start;
    temperature_iteration = number_of_heuristics;
    iter = 0;
    counter_accepted_heuristics = 0;
    reheat = false;
    worse = false;

    // Reset the LLH parameters
    shift_heuristic_instance = {SHIFT, 0, 0, 0, wmin, wmin}; 
    split_heuristic_instance = {SPLIT, 0, 0, 0, wmin, wmin}; 
    largest_bin_heuristic_instance = {LARGEST_BIN, 0, 0, 0, wmin, wmin}; 
    shuffle_heuristic_instance = {SHUFFLE, 0, 0, 0, wmin, wmin}; 
    best_packing_heuristic_instance = {BEST_PACKING, 0, 0, 0, wmin, wmin}; 

    low_level_heuristics = {
        shift_heuristic_instance, 
        split_heuristic_instance, 
        largest_bin_heuristic_instance, 
        shuffle_heuristic_instance, 
        best_packing_heuristic_instance
    };
}

void update_best_solution(Solution solution){
    // initialize best solution for new problem instance
    if (best_sln.objective == 0){
        best_sln = solution;
    }
    // update global best solution if new solution is better
    if (solution.objective < best_sln.objective){
        best_sln = solution;
    }
}

// Simulated Annealing algorithm to optimize solutions for given problems
void SimulatedAnnealing(Problem* problem) {
    // Reset variables to initial states for the new problem
    reset(); 

    Solution candidate_solution = init_solution(problem);           // Initialize the candidate solution
    Solution current_solution = candidate_solution;                 // Set the current solution as the candidate solution

    update_best_solution(candidate_solution);                       // Update the global best solution with the initial candidate

    auto time_start = std::chrono::high_resolution_clock::now();    // Start timing the algorithm
    double time_spent = 0;

    while (iter < SA_MAX_iteration && time_spent < MAX_TIME) {      // Continue until max iterations or max time is reached
        iter++;                                                     // Increment the iteration count
        random_check = rand_01();                                   // Generate a random number for decision making
        
        // Generate a neighboring solution using a weighted low-level heuristic
        candidate_solution = apply_weighted_low_level_heuristic(current_solution);

        // Calculate the change in the objective function
        int delta = candidate_solution.objective - current_solution.objective;

        // Decision making for accepting the new solution
        if (delta <= 0 && !compare_same_solution(candidate_solution, current_solution)) {
            current_solution = candidate_solution;                  // Accept the better solution
        } else if (delta > 0 && (exp((-1 * static_cast<float>(delta)) / static_cast<float>(temperature)) > random_check)) {
            current_solution = candidate_solution;                  // Accept the worse solution
            worse = true;                                           // Flag that a worse solution was accepted
        }

        // Reheat schedule if necessary
        if (reheat) {
            temperature_improvement = temperature_improvement / (1 - (beta * temperature_improvement));
            temperature = temperature_improvement;                  // Update the temperature
        }   
        // Cooling schedule
        else if (iter % temperature_iteration == 0) {
            temperature = temperature / (1 + (beta * temperature)); // Cool down the temperature
        }

        learn(candidate_solution, current_solution, delta);         // Update heuristic learning based on this iteration
        update_best_solution(current_solution);                     // Check if the current solution is the best solution

        auto time_fin = std::chrono::high_resolution_clock::now();  // End timing for this iteration
        time_spent = std::chrono::duration_cast<std::chrono::seconds>(time_fin - time_start).count(); // Calculate total time spent
        if (best_sln.objective == global_best) {
            break;                                                  // Exit if the global best solution is found
        }
    }
}

// ================================================= MAIN ======================================================
int main(int argc, const char * argv[]) {
    int seed = 54;  // Default seed
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-seed" && i + 1 < argc) {
            seed = std::atoi(argv[i+1]);
        }
    }
    init_random_generator(seed);
    std::cout << "Starting the run...\n";
    if(argc<3)
    {
        printf("Insufficient arguments. Please use the following options:\n   -s data_file (compulsory)\n   -o out_file (default my_solutions.txt)\n   -c solution_file_to_check\n   -t max_time (in sec)\n");
        return 1;
    }
    else if(argc>9)
    {
        printf("Too many arguments.\n");
        return 2;
    }
    else
    {
        for(int i=1; i<argc; i=i+2)
        {
            if(strcmp(argv[i],"-s")==0)
                strcpy(data_file, argv[i+1]);
            else if(strcmp(argv[i],"-o")==0)
                strcpy(out_file, argv[i+1]);
            else if(strcmp(argv[i],"-c")==0)
                strcpy(solution_file, argv[i+1]);
            else if(strcmp(argv[i],"-t")==0)
                MAX_TIME = atoi(argv[i+1]);
        }
    }
    std ::cout<< "LOADING PROBLEM...\n";
    // Load problems
    std::vector<Problem*> problems = load_problems(data_file);
    std::cout << "LOADED PROBLEMS. Number of loaded problems: " << num_of_problems << std::endl;

    if(strcmp(out_file,"")==0) strcpy(out_file, "my_solutions.txt");        // default output
    FILE* pfile = fopen(out_file, "w");                                     // open a new file
    fprintf(pfile, "%d\n", num_of_problems); fclose(pfile);

    int gap = 0;

    for(int k=0; k<num_of_problems; k++)
    {
        std::cout << "\n================== NEW PROBLEM: " << problems[k]->problem_identifier << " =====================" << std::endl;
        best_sln.prob = problems[k];
        global_best = problems[k]->best_obj;
        int current_gap = 0;
        int total_gap_per_run = 0;
        float abs_gap = 0.0;
        for(int run=0; run<NUM_OF_RUNS; run++)
        {
            // reset for new run
            best_sln.objective = 0;

            SimulatedAnnealing(problems[k]); // call SA method
            std::cout << "Problem " << best_sln.prob->problem_identifier << ": " << best_sln.objective << ", Best Known Obj: " << global_best;
            current_gap = best_sln.objective - best_sln.prob->best_obj;
            std::cout << ", Gap: " << current_gap << std::endl;
            total_gap_per_run += best_sln.objective;
        }

        // Calculate the avrage objective of single instance
        abs_gap = (total_gap_per_run/ NUM_OF_RUNS) - problems[k]->best_obj;

        // Write the best solution to the output file
        output_solution(best_sln, abs_gap, out_file);
    }

    // Free allocated memory for the problems
    free_problems(problems);
    return 0;
}
