#include <omp.h>
#include <iostream>
#include <string>

#include "loaders.hpp"         // Dataset handling
#include "decision_tree.hpp"   // Decision tree
#include "metrics.hpp"         // Metrics for evaluation
#include "random_forest.hpp"   // Random forest
#include "progress.hpp"        // Progress tracking
#include "benchmark.hpp"       // Benchmark utilities

using namespace std;

// Global flag for progress bar visibility
bool g_show_progress = true;

// Wrapper functions for manual testing
void test_decision_tree(const DatasetConfig& dataset_config, bool use_parallel) {
    run_decision_tree_benchmark(dataset_config, use_parallel, false);
}

void test_random_forest(const DatasetConfig& dataset_config, bool use_forest_parallel, bool use_tree_parallel, int num_trees) {
    run_random_forest_benchmark(dataset_config, use_forest_parallel, use_tree_parallel, num_trees, false);
}

void menu() {
    cout << "\n========================================" << endl;
    cout << "   Parallel Random Forests Demo" << endl;
    cout << "========================================" << endl;
    cout << "\nChoose algorithm:" << endl;
    cout << "1. Random Forest" << endl;
    cout << "2. Decision Tree" << endl;
    cout << "Enter your choice: ";
    
    int testChoice;
    cin >> testChoice;

    cout << "\nChoose dataset:" << endl;
    cout << "1. Diabetes (768 samples, 8 features, binary classification)" << endl;
    cout << "2. Palmer Penguins (344 samples, 5 features, 3 classes)" << endl;
    cout << "3. Dry Bean (13,611 samples, 16 features, 7 classes)" << endl;
    cout << "Enter your choice: ";
    
    int datasetChoice;
    cin >> datasetChoice;

    DatasetConfig dataset_config = get_dataset_config(datasetChoice);

    cout << "\nShow progress bar during training? (y/n): ";
    string showProgressStr;
    cin >> showProgressStr;
    g_show_progress = (showProgressStr == "y" || showProgressStr == "Y");

    cout << "\nChoose mode:" << endl;
    cout << "1. Manual (configure parallelism options)" << endl;
    cout << "2. Benchmark (test all parallelism configurations)" << endl;
    cout << "3. Sample Size Benchmark (test different dataset sizes)" << endl;
    cout << "Enter your choice: ";
    
    int modeChoice;
    cin >> modeChoice;

    if (modeChoice == 1) {
        // MANUAL MODE
        if (testChoice == 1) {
            // Random Forest
            cout << "\n--- Random Forest Configuration ---" << endl;
            
            cout << "Use forest-level parallelism? (y/n): ";
            string useForestParallelStr;
            cin >> useForestParallelStr;
            bool useForestParallel = (useForestParallelStr == "y" || useForestParallelStr == "Y");

            cout << "Use tree-level parallelism? (y/n): ";
            string useTreeParallelStr;
            cin >> useTreeParallelStr;
            bool useTreeParallel = (useTreeParallelStr == "y" || useTreeParallelStr == "Y");
            
            cout << "Enter the number of trees: ";
            int numTrees;
            cin >> numTrees;

            if (numTrees <= 0) {
                cout << "Invalid number of trees! Using default: 100" << endl;
                numTrees = 100;
            }

            cout << "\nStarting Random Forest training..." << endl;
            test_random_forest(dataset_config, useForestParallel, useTreeParallel, numTrees);
            
        } else if (testChoice == 2) {
            // Decision Tree
            cout << "\n--- Decision Tree Configuration ---" << endl;
            
            cout << "Use tree-level parallelism? (y/n): ";
            string useTreeParallelStr;
            cin >> useTreeParallelStr;
            bool useTreeParallel = (useTreeParallelStr == "y" || useTreeParallelStr == "Y");

            cout << "\nStarting Decision Tree training..." << endl;
            test_decision_tree(dataset_config, useTreeParallel);
            
        } else {
            cout << "Invalid choice!" << endl;
            exit(1);
        }
        
    } else if (modeChoice == 2) {
        // BENCHMARK MODE
        if (testChoice == 1) {
            // Random Forest Benchmark
            cout << "\n--- Random Forest Benchmark Configuration ---" << endl;
            cout << "Enter the number of trees: ";
            int numTrees;
            cin >> numTrees;

            if (numTrees <= 0) {
                cout << "Invalid number of trees! Using default: 100" << endl;
                numTrees = 100;
            }

            benchmark_random_forest(dataset_config, numTrees);
            
        } else if (testChoice == 2) {
            // Decision Tree Benchmark
            benchmark_decision_tree(dataset_config);
            
        } else {
            cout << "Invalid choice!" << endl;
            exit(1);
        }
        
    } else if (modeChoice == 3) {
        // SAMPLE SIZE BENCHMARK MODE
        if (testChoice == 1) {
            // Random Forest Sample Size Benchmark
            cout << "\n--- Sample Size Benchmark Configuration ---" << endl;
            cout << "This will test sample sizes: 100, 500, 1500, 3500" << endl;
            cout << "Enter the number of trees: ";
            int numTrees;
            cin >> numTrees;

            if (numTrees <= 0) {
                cout << "Invalid number of trees! Using default: 50" << endl;
                numTrees = 50;
            }

            // Only works with Dry Bean dataset
            if (datasetChoice != 3) {
                cout << "\nWarning: Sample size benchmark is designed for Dry Bean dataset (option 3)." << endl;
                cout << "Results may not be meaningful for other datasets." << endl;
            }

            benchmark_sample_sizes(dataset_config, numTrees);
            
        } else if (testChoice == 2) {
            cout << "Sample size benchmark is only available for Random Forest (option 1)." << endl;
            exit(1);
        } else {
            cout << "Invalid choice!" << endl;
            exit(1);
        }
        
    } else {
        cout << "Invalid mode choice!" << endl;
        exit(1);
    }
    
    cout << "\n========================================" << endl;
    cout << "   Complete!" << endl;
    cout << "========================================" << endl;
}

int main() {
    menu();
    return 0;
}
