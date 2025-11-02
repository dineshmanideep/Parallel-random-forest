#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "decision_tree.hpp"
#include "progress.hpp"
#include <vector>

using namespace std;

// Random forest specific configuration
struct random_forest_config {
    int num_trees = 100;                    // Number of trees in the forest
    double bootstrap_sample_ratio = 1.0;    // Ratio of samples to use (1.0 = 100% of data)
    unsigned int random_seed = 42;          // Seed for reproducible bootstrap sampling
    bool use_parallel = true;               // Enable forest-level parallelism (training and prediction)
};

class random_forest {
private:
    vector<decision_tree> trees;
    int num_classes;  // Number of unique classes (learned during fit)
    
    // Generate bootstrap sample (sampling with replacement)
    vector<size_t> generate_bootstrap_sample(
        size_t n_samples,
        size_t sample_size,
        unsigned int seed
    ) const;
    
public:
    // Shared config pointers across all trees
    tree_hyperparameters* hp_config = nullptr;
    tree_growing_config* growing_config = nullptr;
    random_forest_config* rf_config = nullptr;
    RandomForestProgress* progress_tracker = nullptr;  // Optional progress tracking
    
    // Training
    void fit(
        const data_frame& df,
        const vector<string>& feature_cols,
        const string& target_col
    );
    
    // Prediction via majority voting (parallel)
    vector<int> predict(const data_frame& X) const;
    
    // Prediction probabilities - average across all trees (parallel)
    vector<vector<double>> predict_proba(const data_frame& X) const;
};

#endif // RANDOM_FOREST_H
