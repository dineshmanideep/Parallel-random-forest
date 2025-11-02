/*
Random Forest implementation with parallel training and prediction
*/

#include "random_forest.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <omp.h>

using namespace std;

// Generate bootstrap sample (sampling with replacement)
vector<size_t> random_forest::generate_bootstrap_sample(
    size_t n_samples,
    size_t sample_size,
    unsigned int seed
) const {
    vector<size_t> bootstrap_indices;
    bootstrap_indices.reserve(sample_size);
    
    mt19937 rng(seed);
    uniform_int_distribution<size_t> dist(0, n_samples - 1);
    
    for (size_t i = 0; i < sample_size; ++i) {
        bootstrap_indices.push_back(dist(rng));
    }
    
    return bootstrap_indices;
}

// Training
void random_forest::fit(
    const data_frame& df,
    const vector<string>& feature_cols,
    const string& target_col
) {
    if (!rf_config) {
        throw runtime_error("random_forest_config not set. Set rf_config before calling fit().");
    }
    
    // Determine number of classes from target column
    const col* target_column = df.get_column(target_col);
    if (!target_column) {
        throw invalid_argument("Target column not found: " + target_col);
    }
    
    if (auto str_target = dynamic_cast<const string_col*>(target_column)) {
        if (!str_target->has_encoding()) {
            const_cast<string_col*>(str_target)->fit_encoding();
        }
        num_classes = str_target->num_unique_values();
    } else if (auto int_target = dynamic_cast<const int_col*>(target_column)) {
        const auto& data = int_target->get_data();
        num_classes = *max_element(data.begin(), data.end()) + 1;
    } else {
        throw invalid_argument("Target column must be string or int type");
    }
    
    // Resize trees vector
    int num_trees = rf_config->num_trees;
    trees.resize(num_trees);
    
    // Calculate bootstrap sample size
    size_t n_samples = df.get_num_rows();
    size_t sample_size = static_cast<size_t>(n_samples * rf_config->bootstrap_sample_ratio);
    
    // Generate all bootstrap samples (sequential - fast enough)
    vector<vector<size_t>> bootstrap_samples(num_trees);
    for (int i = 0; i < num_trees; ++i) {
        bootstrap_samples[i] = generate_bootstrap_sample(
            n_samples,
            sample_size,
            rf_config->random_seed + i
        );
    }
    
    // Initialize progress tracker if provided
    if (progress_tracker) {
        progress_tracker->initialize(num_trees);
        
        // Initialize each tree's progress tracker
        int max_d = (hp_config ? hp_config->max_depth : -1);
        int min_samples = (hp_config ? hp_config->min_examples_per_leaf : 1);
        for (int i = 0; i < num_trees; ++i) {
            progress_tracker->initialize_tree(i, max_d, min_samples, sample_size);
        }
    }
    
    // Train all trees (parallel or sequential based on config)
    if (rf_config->use_parallel) {
        #pragma omp parallel for
        for (int i = 0; i < num_trees; ++i) {
            trees[i].hp_config = hp_config;
            trees[i].growing_config = growing_config;
            
            // Link tree to its progress tracker
            if (progress_tracker) {
                trees[i].progress_tracker = &(progress_tracker->tree_progresses[i]);
            }
            
            trees[i].fit(df, feature_cols, target_col, &bootstrap_samples[i]);
            
            // Mark tree complete and update display
            if (progress_tracker) {
                progress_tracker->mark_tree_complete(i);
                progress_tracker->print_progress();
            }
        }
    } else {
        for (int i = 0; i < num_trees; ++i) {
            trees[i].hp_config = hp_config;
            trees[i].growing_config = growing_config;
            
            // Link tree to its progress tracker
            if (progress_tracker) {
                trees[i].progress_tracker = &(progress_tracker->tree_progresses[i]);
            }
            
            trees[i].fit(df, feature_cols, target_col, &bootstrap_samples[i]);
            
            // Mark tree complete and update display
            if (progress_tracker) {
                progress_tracker->mark_tree_complete(i);
                progress_tracker->print_progress();
            }
        }
    }
    
    // Finalize progress display
    if (progress_tracker) {
        progress_tracker->finish();
    }
}

// Prediction via majority voting
vector<int> random_forest::predict(const data_frame& X) const {
    if (trees.empty()) {
        throw runtime_error("Forest not fitted. Call fit() first.");
    }
    
    int num_trees = trees.size();
    size_t n_samples = X.get_num_rows();
    
    // Get predictions from all trees
    vector<vector<int>> all_predictions(num_trees);
    
    if (rf_config && rf_config->use_parallel) {
        #pragma omp parallel for
        for (int i = 0; i < num_trees; ++i) {
            all_predictions[i] = trees[i].predict(X);
        }
    } else {
        for (int i = 0; i < num_trees; ++i) {
            all_predictions[i] = trees[i].predict(X);
        }
    }
    
    // Aggregate via majority voting
    vector<int> final_predictions(n_samples);
    
    if (rf_config && rf_config->use_parallel) {
        #pragma omp parallel for
        for (size_t row = 0; row < n_samples; ++row) {
            vector<int> votes(num_classes, 0);
            
            // Count votes from all trees
            for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
                int prediction = all_predictions[tree_idx][row];
                if (prediction >= 0 && prediction < num_classes) {
                    votes[prediction]++;
                }
            }
            
            // Find class with most votes
            final_predictions[row] = max_element(votes.begin(), votes.end()) - votes.begin();
        }
    } else {
        for (size_t row = 0; row < n_samples; ++row) {
            vector<int> votes(num_classes, 0);
            
            // Count votes from all trees
            for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
                int prediction = all_predictions[tree_idx][row];
                if (prediction >= 0 && prediction < num_classes) {
                    votes[prediction]++;
                }
            }
            
            // Find class with most votes
            final_predictions[row] = max_element(votes.begin(), votes.end()) - votes.begin();
        }
    }
    
    return final_predictions;
}

// Prediction probabilities - average across all trees
vector<vector<double>> random_forest::predict_proba(const data_frame& X) const {
    if (trees.empty()) {
        throw runtime_error("Forest not fitted. Call fit() first.");
    }
    
    int num_trees = trees.size();
    size_t n_samples = X.get_num_rows();
    
    // Get probabilities from all trees
    vector<vector<vector<double>>> all_probabilities(num_trees);
    
    if (rf_config && rf_config->use_parallel) {
        #pragma omp parallel for
        for (int i = 0; i < num_trees; ++i) {
            all_probabilities[i] = trees[i].predict_proba(X);
        }
    } else {
        for (int i = 0; i < num_trees; ++i) {
            all_probabilities[i] = trees[i].predict_proba(X);
        }
    }
    
    // Average probabilities across trees
    vector<vector<double>> final_probabilities(n_samples, vector<double>(num_classes, 0.0));
    
    if (rf_config && rf_config->use_parallel) {
        #pragma omp parallel for
        for (size_t row = 0; row < n_samples; ++row) {
            // Sum probabilities from all trees
            for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
                for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
                    final_probabilities[row][class_idx] += all_probabilities[tree_idx][row][class_idx];
                }
            }
            
            // Average by dividing by number of trees
            for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
                final_probabilities[row][class_idx] /= num_trees;
            }
        }
    } else {
        for (size_t row = 0; row < n_samples; ++row) {
            // Sum probabilities from all trees
            for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
                for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
                    final_probabilities[row][class_idx] += all_probabilities[tree_idx][row][class_idx];
                }
            }
            
            // Average by dividing by number of trees
            for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
                final_probabilities[row][class_idx] /= num_trees;
            }
        }
    }
    
    return final_probabilities;
}

