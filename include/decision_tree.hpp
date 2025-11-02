#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "loaders.hpp"
#include "metrics.hpp"
#include "progress.hpp"

#include <omp.h>
#include <vector>
#include <memory>

using namespace std;

// Hyperparameters controlling tree structure
struct tree_hyperparameters {
    int max_depth = -1;              // -1 = unlimited depth
    int min_examples_per_leaf = 1;   // Minimum samples required to be a leaf
};

// Configuration for tree growing process
struct tree_growing_config {
    enum class SplitCriterion { 
        GINI, 
        SHANNON_ENTROPY
    };
    
    SplitCriterion criterion = SplitCriterion::GINI;
    int max_features_per_split = -1;  // -1 = use all features (for future random forest)
    
    // Parallelism configuration
    bool use_parallel = false;           // Enable tree-level parallelism
    int min_samples_for_parallel = 100;  // Minimum samples in node to spawn parallel tasks
    int max_parallel_depth = 8;          // Maximum depth to spawn tasks (prevents task explosion)
};

class decision_tree {
private:
    // Internal tree node structure
    struct TreeNode {
        bool is_leaf;
        
        // Internal node - split information
        int feature_idx;           // Index into feature_names vector
        bool is_categorical;       // True if split on categorical feature
        
        // For numerical features: feature_value <= threshold
        double threshold;
        
        // For categorical features: feature_value == split_value (one-vs-rest)
        // Extensible: can change to set<string> for subset-based splits later
        string split_value;
        
        unique_ptr<TreeNode> left;   // Left child
        unique_ptr<TreeNode> right;  // Right child
        
        // Leaf node - prediction information
        int predicted_class;              // Class with highest probability
        vector<double> class_probabilities;  // Probability distribution over classes
    };
    
    unique_ptr<TreeNode> root;
    
    // Learned during fit
    vector<string> feature_names;  // Names of features used for training
    string target_column_name;     // Name of target column
    int num_classes;               // Number of unique classes in target
    
    // Helper: Recursively build decision tree
    unique_ptr<TreeNode> build_tree(
        const data_frame& df,
        const vector<size_t>& indices,     // Row indices to use for this node
        int current_depth
    );
    
    // Helper: Find best split for numerical feature
    // Returns: (best_gain, best_threshold)
    pair<double, double> find_best_numerical_split(
        const data_frame& df,
        const vector<size_t>& indices,
        int feature_idx,
        const vector<int>& encoded_labels
    );
    
    // Helper: Find best split for categorical feature (one-vs-rest)
    // Returns: (best_gain, best_split_value)
    pair<double, string> find_best_categorical_split(
        const data_frame& df,
        const vector<size_t>& indices,
        int feature_idx,
        const vector<int>& encoded_labels
    );
    
    // Helper: Traverse tree to predict single sample
    int predict_single(
        const TreeNode* node, 
        const data_frame& X, 
        size_t row_idx
    ) const;
    
    // Helper: Get class probabilities for single sample
    vector<double> predict_proba_single(
        const TreeNode* node,
        const data_frame& X,
        size_t row_idx
    ) const;

public:
    // Public configuration pointers (user sets these manually)
    tree_hyperparameters* hp_config = nullptr;
    tree_growing_config* growing_config = nullptr;
    TreeProgress* progress_tracker = nullptr;  // Optional progress tracking
    
    // Training
    void fit(
        const data_frame& df,
        const vector<string>& feature_cols,  // Names of feature columns to use
        const string& target_col,             // Name of target column
        const vector<size_t>* bootstrap_indices = nullptr  // nullptr = use all rows
    );
    
    // Prediction - returns encoded class labels (0, 1, 2, ...)
    vector<int> predict(const data_frame& X) const;
    
    // Prediction - returns class probability distributions
    vector<vector<double>> predict_proba(const data_frame& X) const;
};

#endif // DECISION_TREE_H