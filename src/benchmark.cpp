#include "benchmark.hpp"
#include "decision_tree.hpp"
#include "random_forest.hpp"
#include "metrics.hpp"
#include "progress.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;

DatasetConfig get_dataset_config(int datasetChoice) {
    DatasetConfig config;
    
    if (datasetChoice == 1) {
        // Diabetes dataset
        config.path = "dataset/diabetes.csv";
        config.feature_cols = {
            "Pregnancies", 
            "Glucose", 
            "BloodPressure", 
            "SkinThickness", 
            "Insulin", 
            "BMI", 
            "DiabetesPedigreeFunction", 
            "Age"
        };
        config.target_col = "Outcome";
        config.needs_encoding = false;  // Numeric target
    } else if (datasetChoice == 2) {
        // Palmer Penguins dataset
        config.path = "dataset/palmer_penguins.csv";
        config.feature_cols = {
            "island", 
            "bill_length_mm", 
            "bill_depth_mm", 
            "flipper_length_mm", 
            "body_mass_g"
        };
        config.target_col = "species";
        config.needs_encoding = true;  // String target needs encoding
    } else if (datasetChoice == 3) {
        // Dry Bean dataset
        config.path = "dataset/Dry_Bean_Dataset.csv";
        config.feature_cols = {
            "Area",
            "Perimeter",
            "MajorAxisLength",
            "MinorAxisLength",
            "AspectRation",
            "Eccentricity",
            "ConvexArea",
            "EquivDiameter",
            "Extent",
            "Solidity",
            "roundness",
            "Compactness",
            "ShapeFactor1",
            "ShapeFactor2",
            "ShapeFactor3",
            "ShapeFactor4"
        };
        config.target_col = "Class";
        config.needs_encoding = true;  // String target needs encoding
    } else {
        cout << "Invalid dataset choice!" << endl;
        exit(1);
    }
    
    return config;
}

void print_benchmark_table(const vector<BenchmarkResult>& results) {
    cout << "\n========================================" << endl;
    cout << "       BENCHMARK RESULTS" << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    // Header
    cout << left;
    cout << setw(30) << "Configuration" 
         << setw(15) << "Time (ms)" 
         << setw(12) << "Speedup" 
         << setw(12) << "Accuracy" 
         << endl;
    cout << string(69, '-') << endl;
    
    // Results
    for (const auto& result : results) {
        cout << setw(30) << result.config_name 
             << setw(15) << fixed << setprecision(2) << result.training_time_ms 
             << setw(12) << fixed << setprecision(2) << result.speedup << "x"
             << setw(12) << fixed << setprecision(4) << result.accuracy 
             << endl;
    }
    
    cout << string(69, '-') << endl;
    cout << endl;
}

BenchmarkResult run_decision_tree_benchmark(const DatasetConfig& dataset_config, bool use_parallel, bool silent) {
    if (!silent) {
        cout << "\n=== Testing Decision Tree ===" << endl;
        cout << "Loading dataset from: " << dataset_config.path << endl;
    }
    
    // Importing dataset
    data_frame df = data_frame::import_from(dataset_config.path);

    // ============================================================
    // DATASET SUBSAMPLING FOR LARGE DATASETS
    // For Dry Bean dataset: Use only 25% of data for faster training
    // Change this ratio (0.25) to use more/less data
    // ============================================================
    if (dataset_config.path == "dataset/Dry_Bean_Dataset.csv") {
        auto [subset_df, _] = df.train_test_split(0.75);  // Keep 25%, discard 75%
        df = std::move(subset_df);
        if (!silent) {
            cout << "Using 25% subset for faster training (approx 3,400 samples)" << endl;
        }
    }

    // Train test split
    auto [train_df, test_df] = df.train_test_split(0.2);

    // Fit encoding if needed
    if (dataset_config.needs_encoding) {
        train_df.get_string_column(dataset_config.target_col)->fit_encoding();
        test_df.get_string_column(dataset_config.target_col)->fit_encoding();
    }

    if (!silent) {
        cout << "Data loaded and encoded. Fitting tree..." << endl;
    }

    // Initializing decision tree
    decision_tree tree;

    // Setting tree growing configuration
    tree_growing_config growing_config;
    growing_config.criterion = tree_growing_config::SplitCriterion::GINI;
    growing_config.max_features_per_split = -1;

    growing_config.use_parallel = use_parallel;
    growing_config.min_samples_for_parallel = 100;
    growing_config.max_parallel_depth = 8;

    tree.growing_config = &growing_config;

    // Setting hyperparameters for regularization
    tree_hyperparameters hp_config;
    hp_config.max_depth = 100;
    hp_config.min_examples_per_leaf = 5;

    tree.hp_config = &hp_config;

    auto time_start = chrono::high_resolution_clock::now();
    
    // Fitting tree to training data
    tree.fit(train_df, dataset_config.feature_cols, dataset_config.target_col);

    if (!silent) {
        cout << "Predicting..." << endl;
    }
    
    // Get predictions for test set
    vector<int> predictions = tree.predict(test_df);
    
    auto time_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    
    // Get true labels for test set
    vector<int> encoded_labels;
    if (dataset_config.needs_encoding) {
        vector<string> labels = ((string_col*) test_df.get_column(dataset_config.target_col))->get_data();
        for(string& label : labels) {
            encoded_labels.push_back(
                test_df.get_string_column(dataset_config.target_col)->encode(label)
            );
        }
    } else {
        encoded_labels = test_df.get_int_column(dataset_config.target_col)->get_data();
    }

    // Calculate metrics
    BenchmarkResult result;
    result.training_time_ms = duration.count();
    result.accuracy = metrics::accuracy(predictions, encoded_labels);
    result.precision = metrics::precision(predictions, encoded_labels);
    result.recall = metrics::recall(predictions, encoded_labels);
    result.f1_score = metrics::f1_score(predictions, encoded_labels);
    result.speedup = 1.0;  // Will be calculated later

    if (!silent) {
        cout << "\n=== Results ===" << endl;
        cout << "Accuracy:  " << result.accuracy << endl;
        cout << "Precision: " << result.precision << endl;
        cout << "Recall:    " << result.recall << endl;
        cout << "F1 score:  " << result.f1_score << endl;
        cout << "Training & evaluation time taken: " << result.training_time_ms << " milliseconds" << endl;
    }

    return result;
}

BenchmarkResult run_random_forest_benchmark(const DatasetConfig& dataset_config, bool use_forest_parallel, bool use_tree_parallel, int num_trees, bool silent) {
    if (!silent) {
        cout << "\n=== Testing Random Forest ===" << endl;
        cout << "Loading dataset from: " << dataset_config.path << endl;
    }

    // Importing dataset
    data_frame df = data_frame::import_from(dataset_config.path);

    // ============================================================
    // DATASET SUBSAMPLING FOR LARGE DATASETS
    // For Dry Bean dataset: Use only 25% of data for faster training
    // Change this ratio (0.25) to use more/less data
    // ============================================================
    if (dataset_config.path == "dataset/Dry_Bean_Dataset.csv") {
        auto [subset_df, _] = df.train_test_split(0.75);  // Keep 25%, discard 75%
        df = std::move(subset_df);
        if (!silent) {
            cout << "Using 25% subset for faster training (approx 3,400 samples)" << endl;
        }
    }

    auto [train_df, test_df] = df.train_test_split(0.2);
    
    // Fit encoding if needed
    if (dataset_config.needs_encoding) {
        train_df.get_string_column(dataset_config.target_col)->fit_encoding();
        test_df.get_string_column(dataset_config.target_col)->fit_encoding();
    }

    if (!silent) {
        cout << "Data loaded. Fitting forest with " << num_trees << " trees..." << endl;
    }

    // Initializing random forest
    random_forest forest;

    // Setting random forest configuration
    random_forest_config config;
    config.num_trees = num_trees;
    config.bootstrap_sample_ratio = 0.55;
    config.use_parallel = use_forest_parallel;

    forest.rf_config = &config;

    // Setting tree growing configuration
    tree_growing_config growing_config;
    growing_config.criterion = tree_growing_config::SplitCriterion::GINI;
    growing_config.max_features_per_split = -1;

    growing_config.use_parallel = use_tree_parallel;
    growing_config.min_samples_for_parallel = 100;
    growing_config.max_parallel_depth = 8;

    forest.growing_config = &growing_config;

    // Setting hyperparameters for regularization
    tree_hyperparameters hp_config;
    hp_config.max_depth = 300;
    hp_config.min_examples_per_leaf = 20;

    forest.hp_config = &hp_config;

    // Enable progress tracking only in non-silent mode and if globally enabled
    RandomForestProgress progress;
    if (!silent && g_show_progress) {
        forest.progress_tracker = &progress;
    }

    // Fitting forest to training data
    auto time_start = chrono::high_resolution_clock::now();
    forest.fit(train_df, dataset_config.feature_cols, dataset_config.target_col);

    // Getting predictions for test set
    vector<int> predictions = forest.predict(test_df);
    
    auto time_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    
    // Getting true labels for test set
    vector<int> encoded_labels;
    if (dataset_config.needs_encoding) {
        vector<string> labels = ((string_col*) test_df.get_column(dataset_config.target_col))->get_data();
        for(string& label : labels) {
            encoded_labels.push_back(test_df.get_string_column(dataset_config.target_col)->encode(label));
        }
    } else {
        encoded_labels = test_df.get_int_column(dataset_config.target_col)->get_data();
    }

    // Calculate metrics
    BenchmarkResult result;
    result.training_time_ms = duration.count();
    result.accuracy = metrics::accuracy(predictions, encoded_labels);
    result.precision = metrics::precision(predictions, encoded_labels);
    result.recall = metrics::recall(predictions, encoded_labels);
    result.f1_score = metrics::f1_score(predictions, encoded_labels);
    result.speedup = 1.0;  // Will be calculated later

    if (!silent) {
        cout << "\n=== Results ===" << endl;
        cout << "Accuracy:  " << result.accuracy << endl;
        cout << "Precision: " << result.precision << endl;
        cout << "Recall:    " << result.recall << endl;
        cout << "F1 score:  " << result.f1_score << endl;
        cout << "Training & evaluation time taken: " << result.training_time_ms << " milliseconds" << endl;
    }

    return result;
}

void benchmark_decision_tree(const DatasetConfig& dataset_config) {
    cout << "\n=== BENCHMARKING DECISION TREE ===" << endl;
    cout << "Running tests with different parallelism configurations...\n" << endl;
    
    vector<BenchmarkResult> results;
    
    // Test 1: Fully Serial
    cout << "[1/2] Testing fully serial version..." << endl;
    BenchmarkResult serial_result = run_decision_tree_benchmark(dataset_config, false, false);
    serial_result.config_name = "Serial (No Parallelism)";
    serial_result.speedup = 1.0;
    results.push_back(serial_result);
    
    // Test 2: Tree-level Parallelism
    cout << "[2/2] Testing with tree-level parallelism..." << endl;
    BenchmarkResult tree_parallel_result = run_decision_tree_benchmark(dataset_config, true, false);
    tree_parallel_result.config_name = "Tree-level Parallelism";
    tree_parallel_result.speedup = serial_result.training_time_ms / tree_parallel_result.training_time_ms;
    results.push_back(tree_parallel_result);
    
    // Print results table
    print_benchmark_table(results);
}

void benchmark_random_forest(const DatasetConfig& dataset_config, int num_trees) {
    cout << "\n=== BENCHMARKING RANDOM FOREST ===" << endl;
    cout << "Running tests with different parallelism configurations..." << endl;
    cout << "Number of trees: " << num_trees << endl;
    cout << "Note: Large datasets may take several minutes per configuration.\n" << endl;
    
    vector<BenchmarkResult> results;
    
    // Test 1: Fully Serial
    cout << "[1/3] Testing fully serial version..." << endl;
    BenchmarkResult serial_result = run_random_forest_benchmark(dataset_config, false, false, num_trees, false);
    serial_result.config_name = "Serial (No Parallelism)";
    serial_result.speedup = 1.0;
    results.push_back(serial_result);
    
    // Test 2: Tree-level Parallelism Only
    cout << "[2/3] Testing with tree-level parallelism..." << endl;
    BenchmarkResult tree_parallel_result = run_random_forest_benchmark(dataset_config, false, true, num_trees, false);
    tree_parallel_result.config_name = "Tree-level Parallelism";
    tree_parallel_result.speedup = serial_result.training_time_ms / tree_parallel_result.training_time_ms;
    results.push_back(tree_parallel_result);
    
    // Test 3: Forest-level Parallelism Only
    cout << "[3/3] Testing with forest-level parallelism..." << endl;
    BenchmarkResult forest_parallel_result = run_random_forest_benchmark(dataset_config, true, false, num_trees, false);
    forest_parallel_result.config_name = "Forest-level Parallelism";
    forest_parallel_result.speedup = serial_result.training_time_ms / forest_parallel_result.training_time_ms;
    results.push_back(forest_parallel_result);
    
    // Print results table
    print_benchmark_table(results);
}

// Helper function to run benchmark with specific sample size
BenchmarkResult run_benchmark_with_sample_size(const DatasetConfig& dataset_config, int target_samples, int num_trees, bool use_forest_parallel, bool use_tree_parallel) {
    // Load full dataset
    data_frame df = data_frame::import_from(dataset_config.path);
    
    // Calculate the split ratio to get approximately target_samples
    // For Dry Bean: 13611 total samples
    // We want target_samples, so keep ratio = target_samples / 13611
    double total_samples = 13611.0;  // Dry Bean dataset size
    double keep_ratio = target_samples / total_samples;
    double discard_ratio = 1.0 - keep_ratio;
    
    // Subsample to get target size
    auto [subset_df, _] = df.train_test_split(discard_ratio);
    df = std::move(subset_df);
    
    // Train-test split (80-20)
    auto [train_df, test_df] = df.train_test_split(0.2);
    
    // Fit encoding if needed
    if (dataset_config.needs_encoding) {
        train_df.get_string_column(dataset_config.target_col)->fit_encoding();
        test_df.get_string_column(dataset_config.target_col)->fit_encoding();
    }
    
    // Initialize random forest
    random_forest forest;
    
    // Configure random forest
    random_forest_config config;
    config.num_trees = num_trees;
    config.bootstrap_sample_ratio = 0.55;
    config.use_parallel = use_forest_parallel;
    forest.rf_config = &config;
    
    // Configure tree growing
    tree_growing_config growing_config;
    growing_config.criterion = tree_growing_config::SplitCriterion::GINI;
    growing_config.max_features_per_split = -1;
    growing_config.use_parallel = use_tree_parallel;
    growing_config.min_samples_for_parallel = 100;
    growing_config.max_parallel_depth = 8;
    forest.growing_config = &growing_config;
    
    // Configure hyperparameters
    tree_hyperparameters hp_config;
    hp_config.max_depth = 300;
    hp_config.min_examples_per_leaf = 20;
    forest.hp_config = &hp_config;
    
    // Fit and measure time
    auto time_start = chrono::high_resolution_clock::now();
    forest.fit(train_df, dataset_config.feature_cols, dataset_config.target_col);
    vector<int> predictions = forest.predict(test_df);
    auto time_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    
    // Get true labels
    vector<int> encoded_labels;
    if (dataset_config.needs_encoding) {
        vector<string> labels = ((string_col*) test_df.get_column(dataset_config.target_col))->get_data();
        for(string& label : labels) {
            encoded_labels.push_back(test_df.get_string_column(dataset_config.target_col)->encode(label));
        }
    } else {
        encoded_labels = test_df.get_int_column(dataset_config.target_col)->get_data();
    }
    
    // Calculate metrics
    BenchmarkResult result;
    result.training_time_ms = duration.count();
    result.accuracy = metrics::accuracy(predictions, encoded_labels);
    result.precision = metrics::precision(predictions, encoded_labels);
    result.recall = metrics::recall(predictions, encoded_labels);
    result.f1_score = metrics::f1_score(predictions, encoded_labels);
    result.speedup = 1.0;
    
    return result;
}

void benchmark_sample_sizes(const DatasetConfig& dataset_config, int num_trees) {
    cout << "\n=== BENCHMARKING DIFFERENT SAMPLE SIZES ===" << endl;
    cout << "Comparing serial vs parallel performance across different dataset sizes" << endl;
    cout << "Number of trees: " << num_trees << "\n" << endl;
    
    // Sample sizes to test: 100, 500, 1500, 3500
    vector<int> sample_sizes = {100, 500, 1500, 3500};
    vector<BenchmarkResult> results;
    
    for (size_t i = 0; i < sample_sizes.size(); i++) {
        int samples = sample_sizes[i];
        cout << "\n[" << (i+1) << "/" << sample_sizes.size() << "] Testing with " << samples << " samples..." << endl;
        
        // Test 1: Serial version
        cout << "  -> Running serial version..." << endl;
        BenchmarkResult serial_result = run_benchmark_with_sample_size(dataset_config, samples, num_trees, false, false);
        serial_result.config_name = to_string(samples) + " samples (Serial)";
        serial_result.speedup = 1.0;  // Baseline
        results.push_back(serial_result);
        
        // Test 2: Forest-parallel version
        cout << "  -> Running forest-parallel version..." << endl;
        BenchmarkResult parallel_result = run_benchmark_with_sample_size(dataset_config, samples, num_trees, true, false);
        parallel_result.config_name = to_string(samples) + " samples (Forest-parallel)";
        parallel_result.speedup = serial_result.training_time_ms / parallel_result.training_time_ms;
        results.push_back(parallel_result);
    }
    
    // Print results table
    print_benchmark_table(results);
    
    // Additional analysis: Extract serial and parallel times
    cout << "\n=== Sample Size Scaling Analysis ===" << endl;
    cout << "\nSerial Performance:" << endl;
    cout << "Sample Size | Time (ms) | Time Ratio" << endl;
    cout << string(45, '-') << endl;
    
    double serial_baseline = results[0].training_time_ms;
    for (size_t i = 0; i < sample_sizes.size(); i++) {
        int idx = i * 2;  // Serial results are at even indices
        double time_ratio = results[idx].training_time_ms / serial_baseline;
        cout << setw(11) << sample_sizes[i] << " | " 
             << setw(9) << fixed << setprecision(2) << results[idx].training_time_ms << " | "
             << setw(10) << fixed << setprecision(2) << time_ratio << "x" << endl;
    }
    
    cout << "\n\nParallel Performance & Speedup:" << endl;
    cout << "Sample Size | Time (ms) | Speedup vs Serial" << endl;
    cout << string(50, '-') << endl;
    
    for (size_t i = 0; i < sample_sizes.size(); i++) {
        int serial_idx = i * 2;
        int parallel_idx = i * 2 + 1;
        cout << setw(11) << sample_sizes[i] << " | " 
             << setw(9) << fixed << setprecision(2) << results[parallel_idx].training_time_ms << " | "
             << setw(17) << fixed << setprecision(2) << results[parallel_idx].speedup << "x" << endl;
    }
    cout << endl;
}

