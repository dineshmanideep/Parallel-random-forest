/*
Serial Decision Tree implementation
*/

#include "decision_tree.hpp"
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <set>
#include <omp.h>

using namespace std;

// Helper: Determine if we should parallelize at this node
static bool should_parallelize(
    int current_depth,
    size_t n_samples,
    const tree_growing_config* config
) {
    if (!config || !config->use_parallel) return false;
    
    // Don't parallelize if config values are invalid
    if (config->max_parallel_depth <= 0 || config->min_samples_for_parallel <= 0) {
        return false;
    }
    
    // Stop if too deep (prevents task explosion)
    if (current_depth >= config->max_parallel_depth) return false;
    
    // Stop if node too small (overhead > benefit)
    if ((int)n_samples < config->min_samples_for_parallel) return false;
    
    return true;
}

// ==================== Training ====================

void decision_tree::fit(
    const data_frame& df,
    const vector<string>& feature_cols,
    const string& target_col,
    const vector<size_t>* bootstrap_indices
) {
    feature_names = feature_cols;
    target_column_name = target_col;
    
    // Prepare target column and determine num_classes
    const col* target_column = df.get_column(target_col);
    if (!target_column) {
        throw invalid_argument("Target column not found: " + target_col);
    }
    
    // Handle string targets - fit encoding
    if (auto str_target = dynamic_cast<const string_col*>(target_column)) {
        if (!str_target->has_encoding()) {
            str_target->fit_encoding();
        }
        num_classes = str_target->num_unique_values();
    } else if (auto int_target = dynamic_cast<const int_col*>(target_column)) {
        // For int targets, find max value
        const auto& data = int_target->get_data();
        num_classes = *max_element(data.begin(), data.end()) + 1;
    } else {
        throw invalid_argument("Target column must be string or int type");
    }
    
    // Set up indices
    vector<size_t> indices;
    if (bootstrap_indices == nullptr) {
        indices.resize(df.get_num_rows());
        iota(indices.begin(), indices.end(), 0);
    } else {
        indices = *bootstrap_indices;
    }
    
    // Initialize progress tracker if provided
    if (progress_tracker) {
        int max_d = (hp_config ? hp_config->max_depth : -1);
        int min_samples = (hp_config ? hp_config->min_examples_per_leaf : 1);
        progress_tracker->initialize(max_d, min_samples, indices.size());
    }
    
    // Build tree recursively
    if (growing_config && growing_config->use_parallel) {
        // Create parallel region for task-based parallelism
        #pragma omp parallel
        {
            #pragma omp single
            {
                root = build_tree(df, indices, 0);
            }
        }
    } else {
        // Sequential execution
        root = build_tree(df, indices, 0);
    }
    
    // Mark progress as complete
    if (progress_tracker) {
        progress_tracker->mark_complete();
    }
}

// ==================== Tree Building ====================

unique_ptr<decision_tree::TreeNode> decision_tree::build_tree(
    const data_frame& df,
    const vector<size_t>& indices,
    int current_depth
) {
    auto node = make_unique<TreeNode>();
    
    // Track node creation
    if (progress_tracker) {
        progress_tracker->increment_nodes();
    }
    
    // Get encoded labels for these indices
    const col* target_col = df.get_column(target_column_name);
    vector<int> encoded_labels;
    
    if (auto str_target = dynamic_cast<const string_col*>(target_col)) {
        for (size_t idx : indices) {
            encoded_labels.push_back(str_target->get_encoded(idx));
        }
    } else if (auto int_target = dynamic_cast<const int_col*>(target_col)) {
        const auto& data = int_target->get_data();
        for (size_t idx : indices) {
            encoded_labels.push_back(data[idx]);
        }
    }
    
    // Check stopping conditions
    
    // 1. Check if node is pure (all same class)
    bool is_pure = true;
    int first_label = encoded_labels[0];
    for (int label : encoded_labels) {
        if (label != first_label) {
            is_pure = false;
            break;
        }
    }
    
    // 2. Check max depth
    bool max_depth_reached = (hp_config != nullptr && 
                              hp_config->max_depth != -1 && 
                              current_depth >= hp_config->max_depth);
    
    // 3. Check min samples
    bool min_samples_reached = (hp_config != nullptr && 
                                (int)indices.size() <= hp_config->min_examples_per_leaf);
    
    // If stopping condition met, create leaf
    if (is_pure || max_depth_reached || min_samples_reached || indices.size() == 1) {
        node->is_leaf = true;
        
        // Calculate class probabilities
        vector<int> counts = metrics::class_counts(encoded_labels, num_classes);
        node->class_probabilities.resize(num_classes);
        for (int c = 0; c < num_classes; ++c) {
            node->class_probabilities[c] = static_cast<double>(counts[c]) / encoded_labels.size();
        }
        
        // Set predicted class (majority)
        node->predicted_class = max_element(counts.begin(), counts.end()) - counts.begin();
        
        return node;
    }
    
    // Find best split across all features
    double best_overall_gain = -numeric_limits<double>::infinity();
    int best_feature_idx = -1;
    bool best_is_categorical = false;
    double best_threshold = 0.0;
    string best_split_value;
    
    for (int feat_idx = 0; feat_idx < (int)feature_names.size(); ++feat_idx) {
        const string& feat_name = feature_names[feat_idx];
        const col* feat_col = df.get_column(feat_name);
        
        if (dynamic_cast<const string_col*>(feat_col)) {
            // Categorical feature
            auto [gain, split_val] = find_best_categorical_split(df, indices, feat_idx, encoded_labels);
            if (gain > best_overall_gain) {
                best_overall_gain = gain;
                best_feature_idx = feat_idx;
                best_is_categorical = true;
                best_split_value = split_val;
            }
        } else {
            // Numerical feature
            auto [gain, threshold] = find_best_numerical_split(df, indices, feat_idx, encoded_labels);
            if (gain > best_overall_gain) {
                best_overall_gain = gain;
                best_feature_idx = feat_idx;
                best_is_categorical = false;
                best_threshold = threshold;
            }
        }
    }
    
    // If no valid split found, create leaf
    if (best_feature_idx == -1 || best_overall_gain <= 0.0) {
        node->is_leaf = true;
        vector<int> counts = metrics::class_counts(encoded_labels, num_classes);
        node->class_probabilities.resize(num_classes);
        for (int c = 0; c < num_classes; ++c) {
            node->class_probabilities[c] = static_cast<double>(counts[c]) / encoded_labels.size();
        }
        node->predicted_class = max_element(counts.begin(), counts.end()) - counts.begin();
        return node;
    }
    
    // Create internal node with best split
    node->is_leaf = false;
    node->feature_idx = best_feature_idx;
    node->is_categorical = best_is_categorical;
    
    if (best_is_categorical) {
        node->split_value = best_split_value;
    } else {
        node->threshold = best_threshold;
    }
    
    // Split indices
    vector<size_t> left_indices, right_indices;
    const string& split_feat_name = feature_names[best_feature_idx];
    const col* split_feat_col = df.get_column(split_feat_name);
    
    if (best_is_categorical) {
        const string_col* str_col = dynamic_cast<const string_col*>(split_feat_col);
        const auto& data = str_col->get_data();
        for (size_t idx : indices) {
            if (data[idx] == best_split_value) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }
    } else {
        if (auto int_col_ptr = dynamic_cast<const int_col*>(split_feat_col)) {
            const auto& data = int_col_ptr->get_data();
            for (size_t idx : indices) {
                if (static_cast<double>(data[idx]) <= best_threshold) {
                    left_indices.push_back(idx);
                } else {
                    right_indices.push_back(idx);
                }
            }
        } else if (auto float_col_ptr = dynamic_cast<const float_col*>(split_feat_col)) {
            const auto& data = float_col_ptr->get_data();
            for (size_t idx : indices) {
                if (data[idx] <= best_threshold) {
                    left_indices.push_back(idx);
                } else {
                    right_indices.push_back(idx);
                }
            }
        }
    }
    
    // Recursively build left and right subtrees
    bool parallelize = should_parallelize(current_depth, indices.size(), growing_config);
    
    if (parallelize) {
        // Parallel task-based execution
        unique_ptr<TreeNode> left_child, right_child;
        
        #pragma omp task shared(left_child, df) firstprivate(left_indices, current_depth) if(growing_config->use_parallel)
        {
            if (!left_indices.empty()) {
                left_child = build_tree(df, left_indices, current_depth + 1);
            }
        }
        
        #pragma omp task shared(right_child, df) firstprivate(right_indices, current_depth) if(growing_config->use_parallel)
        {
            if (!right_indices.empty()) {
                right_child = build_tree(df, right_indices, current_depth + 1);
            }
        }
        
        #pragma omp taskwait  // Wait for both tasks to complete
        
        node->left = move(left_child);
        node->right = move(right_child);
        
    } else {
        // Sequential execution
        if (!left_indices.empty()) {
            node->left = build_tree(df, left_indices, current_depth + 1);
        }
        if (!right_indices.empty()) {
            node->right = build_tree(df, right_indices, current_depth + 1);
        }
    }
    
    return node;
}

pair<double, double> decision_tree::find_best_numerical_split(
    const data_frame& df,
    const vector<size_t>& indices,
    int feature_idx,
    const vector<int>& encoded_labels
) {
    const string& feature_name = feature_names[feature_idx];
    
    // Get feature column
    const col* feature_col = df.get_column(feature_name);
    
    // Collect feature values and labels
    vector<pair<double, int>> values_and_labels;
    
    if (auto int_feat = dynamic_cast<const int_col*>(feature_col)) {
        const auto& data = int_feat->get_data();
        for (size_t i = 0; i < indices.size(); ++i) {
            values_and_labels.push_back({static_cast<double>(data[indices[i]]), encoded_labels[i]});
        }
    } else if (auto float_feat = dynamic_cast<const float_col*>(feature_col)) {
        const auto& data = float_feat->get_data();
        for (size_t i = 0; i < indices.size(); ++i) {
            values_and_labels.push_back({data[indices[i]], encoded_labels[i]});
        }
    } else {
        throw runtime_error("Expected numerical column for numerical split");
    }
    
    // Sort by feature value
    sort(values_and_labels.begin(), values_and_labels.end());
    
    double best_gain = -numeric_limits<double>::infinity();
    double best_threshold = 0.0;
    
    // Collect all labels for parent
    vector<int> parent_labels;
    for (const auto& p : values_and_labels) {
        parent_labels.push_back(p.second);
    }
    
    // Try each possible split point
    for (size_t i = 0; i < values_and_labels.size() - 1; ++i) {
        // Skip if same value
        if (values_and_labels[i].first == values_and_labels[i + 1].first) {
            continue;
        }
        
        double threshold = (values_and_labels[i].first + values_and_labels[i + 1].first) / 2.0;
        
        // Split labels
        vector<int> left_labels, right_labels;
        for (size_t j = 0; j <= i; ++j) {
            left_labels.push_back(values_and_labels[j].second);
        }
        for (size_t j = i + 1; j < values_and_labels.size(); ++j) {
            right_labels.push_back(values_and_labels[j].second);
        }
        
        // Calculate gain based on criterion
        double gain = 0.0;
        if (growing_config && growing_config->criterion == tree_growing_config::SplitCriterion::GINI) {
            gain = metrics::gini_gain(parent_labels, left_labels, right_labels, num_classes);
        } else { // SHANNON_ENTROPY or no config
            gain = metrics::entropy_gain(parent_labels, left_labels, right_labels, num_classes);
        }
        
        if (gain > best_gain) {
            best_gain = gain;
            best_threshold = threshold;
        }
    }
    
    return {best_gain, best_threshold};
}

pair<double, string> decision_tree::find_best_categorical_split(
    const data_frame& df,
    const vector<size_t>& indices,
    int feature_idx,
    const vector<int>& encoded_labels
) {
    const string& feature_name = feature_names[feature_idx];
    const string_col* feature_col = df.get_string_column(feature_name);
    const auto& data = feature_col->get_data();
    
    // Get unique values
    set<string> unique_values;
    for (size_t idx : indices) {
        unique_values.insert(data[idx]);
    }
    
    double best_gain = -numeric_limits<double>::infinity();
    string best_value;
    
    // Collect parent labels (encoded_labels is indexed by position in indices, not absolute index)
    vector<int> parent_labels = encoded_labels;
    
    // Try each unique value as split (one-vs-rest)
    for (const string& split_val : unique_values) {
        vector<int> left_labels, right_labels;
        
        for (size_t i = 0; i < indices.size(); ++i) {
            size_t idx = indices[i];
            if (data[idx] == split_val) {
                left_labels.push_back(encoded_labels[i]);
            } else {
                right_labels.push_back(encoded_labels[i]);
            }
        }
        
        // Skip if split doesn't divide
        if (left_labels.empty() || right_labels.empty()) continue;
        
        // Calculate gain
        double gain = 0.0;
        if (growing_config && growing_config->criterion == tree_growing_config::SplitCriterion::GINI) {
            gain = metrics::gini_gain(parent_labels, left_labels, right_labels, num_classes);
        } else { // SHANNON_ENTROPY or no config
            gain = metrics::entropy_gain(parent_labels, left_labels, right_labels, num_classes);
        }
        
        if (gain > best_gain) {
            best_gain = gain;
            best_value = split_val;
        }
    }
    
    return {best_gain, best_value};
}

// ==================== Prediction ====================

int decision_tree::predict_single(
    const TreeNode* node,
    const data_frame& X,
    size_t row_idx
) const {
    if (node->is_leaf) {
        return node->predicted_class;
    }
    
    const string& feature_name = feature_names[node->feature_idx];
    const col* feature_col = X.get_column(feature_name);
    
    bool go_left = false;
    
    if (node->is_categorical) {
        const string_col* str_col = dynamic_cast<const string_col*>(feature_col);
        go_left = (str_col->get(row_idx) == node->split_value);
    } else {
        double value = 0.0;
        if (auto int_col_ptr = dynamic_cast<const int_col*>(feature_col)) {
            value = static_cast<double>(int_col_ptr->get(row_idx));
        } else if (auto float_col_ptr = dynamic_cast<const float_col*>(feature_col)) {
            value = float_col_ptr->get(row_idx);
        }
        go_left = (value <= node->threshold);
    }
    
    return predict_single(go_left ? node->left.get() : node->right.get(), X, row_idx);
}

vector<double> decision_tree::predict_proba_single(
    const TreeNode* node,
    const data_frame& X,
    size_t row_idx
) const {
    if (node->is_leaf) {
        return node->class_probabilities;
    }
    
    const string& feature_name = feature_names[node->feature_idx];
    const col* feature_col = X.get_column(feature_name);
    
    bool go_left = false;
    
    if (node->is_categorical) {
        const string_col* str_col = dynamic_cast<const string_col*>(feature_col);
        go_left = (str_col->get(row_idx) == node->split_value);
    } else {
        double value = 0.0;
        if (auto int_col_ptr = dynamic_cast<const int_col*>(feature_col)) {
            value = static_cast<double>(int_col_ptr->get(row_idx));
        } else if (auto float_col_ptr = dynamic_cast<const float_col*>(feature_col)) {
            value = float_col_ptr->get(row_idx);
        }
        go_left = (value <= node->threshold);
    }
    
    return predict_proba_single(go_left ? node->left.get() : node->right.get(), X, row_idx);
}

vector<int> decision_tree::predict(const data_frame& X) const {
    if (!root) {
        throw runtime_error("Tree not fitted. Call fit() first.");
    }
    
    vector<int> predictions;
    for (size_t i = 0; i < X.get_num_rows(); ++i) {
        predictions.push_back(predict_single(root.get(), X, i));
    }
    return predictions;
}

vector<vector<double>> decision_tree::predict_proba(const data_frame& X) const {
    if (!root) {
        throw runtime_error("Tree not fitted. Call fit() first.");
    }
    
    vector<vector<double>> probabilities;
    for (size_t i = 0; i < X.get_num_rows(); ++i) {
        probabilities.push_back(predict_proba_single(root.get(), X, i));
    }
    return probabilities;
}
