#include "metrics.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace std;

// Helper: compute class distribution
vector<int> metrics::class_counts(const vector<int>& labels, int num_classes) {
    vector<int> counts(num_classes, 0);
    for (int label : labels) {
        if (label >= 0 && label < num_classes) {
            counts[label]++;
        }
    }
    return counts;
}

// Gini impurity
double metrics::gini_impurity(const vector<int>& labels, int num_classes) {
    if (labels.empty()) return 0.0;
    
    vector<int> counts = class_counts(labels, num_classes);
    double impurity = 1.0;
    int total = labels.size();
    
    for (int count : counts) {
        if (count > 0) {
            double prob = static_cast<double>(count) / total;
            impurity -= prob * prob;
        }
    }
    
    return impurity;
}

// Shannon entropy
double metrics::shannon_entropy(const vector<int>& labels, int num_classes) {
    if (labels.empty()) return 0.0;
    
    vector<int> counts = class_counts(labels, num_classes);
    double entropy = 0.0;
    int total = labels.size();
    
    for (int count : counts) {
        if (count > 0) {
            double prob = static_cast<double>(count) / total;
            entropy -= prob * log2(prob);
        }
    }
    
    return entropy;
}

// Gini gain
double metrics::gini_gain(
    const vector<int>& parent_labels,
    const vector<int>& left_labels,
    const vector<int>& right_labels,
    int num_classes
) {
    if (parent_labels.empty()) return 0.0;
    
    double parent_impurity = gini_impurity(parent_labels, num_classes);
    
    int n_parent = parent_labels.size();
    int n_left = left_labels.size();
    int n_right = right_labels.size();
    
    double left_weight = static_cast<double>(n_left) / n_parent;
    double right_weight = static_cast<double>(n_right) / n_parent;
    
    double weighted_impurity = 
        left_weight * gini_impurity(left_labels, num_classes) +
        right_weight * gini_impurity(right_labels, num_classes);
    
    return parent_impurity - weighted_impurity;
}

// Entropy gain
double metrics::entropy_gain(
    const vector<int>& parent_labels,
    const vector<int>& left_labels,
    const vector<int>& right_labels,
    int num_classes
) {
    if (parent_labels.empty()) return 0.0;
    
    double parent_entropy = shannon_entropy(parent_labels, num_classes);
    
    int n_parent = parent_labels.size();
    int n_left = left_labels.size();
    int n_right = right_labels.size();
    
    double left_weight = static_cast<double>(n_left) / n_parent;
    double right_weight = static_cast<double>(n_right) / n_parent;
    
    double weighted_entropy = 
        left_weight * shannon_entropy(left_labels, num_classes) +
        right_weight * shannon_entropy(right_labels, num_classes);
    
    return parent_entropy - weighted_entropy;
}

// Accuracy
double metrics::accuracy(const vector<int>& predictions, const vector<int>& labels) {
    if (predictions.size() != labels.size() || predictions.empty()) {
        throw invalid_argument("predictions and labels must have same non-zero size");
    }
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / predictions.size();
}

// Precision (macro-averaged for multi-class)
double metrics::precision(const vector<int>& predictions, const vector<int>& labels) {
    if (predictions.size() != labels.size() || predictions.empty()) {
        throw invalid_argument("predictions and labels must have same non-zero size");
    }
    
    int num_classes = *max_element(labels.begin(), labels.end()) + 1;
    double total_precision = 0.0;
    int valid_classes = 0;
    
    for (int c = 0; c < num_classes; ++c) {
        int tp = 0, fp = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == c) {
                if (labels[i] == c) tp++;
                else fp++;
            }
        }
        
        if (tp + fp > 0) {
            total_precision += static_cast<double>(tp) / (tp + fp);
            valid_classes++;
        }
    }
    
    return valid_classes > 0 ? total_precision / valid_classes : 0.0;
}

// Recall (macro-averaged for multi-class)
double metrics::recall(const vector<int>& predictions, const vector<int>& labels) {
    if (predictions.size() != labels.size() || predictions.empty()) {
        throw invalid_argument("predictions and labels must have same non-zero size");
    }
    
    int num_classes = *max_element(labels.begin(), labels.end()) + 1;
    double total_recall = 0.0;
    int valid_classes = 0;
    
    for (int c = 0; c < num_classes; ++c) {
        int tp = 0, fn = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (labels[i] == c) {
                if (predictions[i] == c) tp++;
                else fn++;
            }
        }
        
        if (tp + fn > 0) {
            total_recall += static_cast<double>(tp) / (tp + fn);
            valid_classes++;
        }
    }
    
    return valid_classes > 0 ? total_recall / valid_classes : 0.0;
}

// F1 Score
double metrics::f1_score(const vector<int>& predictions, const vector<int>& labels) {
    double prec = precision(predictions, labels);
    double rec = recall(predictions, labels);
    
    if (prec + rec == 0.0) return 0.0;
    
    return 2.0 * (prec * rec) / (prec + rec);
}
