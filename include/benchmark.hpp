#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include <string>
#include <vector>
#include "loaders.hpp"

// Global flag for progress bar visibility (defined in main.cpp)
extern bool g_show_progress;

struct DatasetConfig {
    std::string path;
    std::vector<std::string> feature_cols;
    std::string target_col;
    bool needs_encoding;
};

struct BenchmarkResult {
    std::string config_name;
    double training_time_ms;
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    double speedup;
};

// Forward declarations
class decision_tree;
class random_forest;

// Benchmark runner functions
BenchmarkResult run_decision_tree_benchmark(
    const DatasetConfig& dataset_config, 
    bool use_parallel, 
    bool silent = false
);

BenchmarkResult run_random_forest_benchmark(
    const DatasetConfig& dataset_config, 
    bool use_forest_parallel, 
    bool use_tree_parallel, 
    int num_trees, 
    bool silent = false
);

// Benchmark suite functions
void benchmark_decision_tree(const DatasetConfig& dataset_config);
void benchmark_random_forest(const DatasetConfig& dataset_config, int num_trees);
void benchmark_sample_sizes(const DatasetConfig& dataset_config, int num_trees);

// Utility functions
void print_benchmark_table(const std::vector<BenchmarkResult>& results);
DatasetConfig get_dataset_config(int datasetChoice);

#endif // BENCHMARK_HPP

