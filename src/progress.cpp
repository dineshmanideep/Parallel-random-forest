/*
Progress tracking implementation for decision trees and random forests
*/

#include "progress.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace std;

// ==================== TreeProgress Implementation ====================

TreeProgress::TreeProgress() : nodes_created(0), estimated_total_nodes(100) {}

void TreeProgress::initialize(int max_depth, int min_samples_per_leaf, size_t n_samples) {
    nodes_created = 0;
    
    if (max_depth == -1) {
        // Unlimited depth - use sample-based estimate
        // Heuristic: roughly n_samples / min_samples_per_leaf leaf nodes
        // Total nodes ≈ 2 * num_leaves (since binary tree)
        int estimated_leaves = max(1, (int)(n_samples / max(1, min_samples_per_leaf)));
        estimated_total_nodes = max(10, 2 * estimated_leaves);
    } else {
        // Use depth-based estimate, but be generous for unbalanced trees
        // For balanced tree: 2^(d+1) - 1 nodes
        // For unbalanced trees (one-vs-rest categorical): multiply by factor
        int balanced_estimate = (1 << (max_depth + 1)) - 1;  // 2^(d+1) - 1
        estimated_total_nodes = max(10, (int)(balanced_estimate * 1.5));  // 50% buffer for imbalance
    }
}

void TreeProgress::increment_nodes() {
    nodes_created++;
}

double TreeProgress::get_progress() const {
    int current = nodes_created.load();
    // Cap at 99% until tree is complete (avoid showing 100% prematurely)
    return min(0.99, current / (double)estimated_total_nodes);
}

void TreeProgress::mark_complete() {
    // Set to estimated total to show 100%
    nodes_created = estimated_total_nodes;
}

// ==================== RandomForestProgress Implementation ====================

RandomForestProgress::RandomForestProgress() 
    : trees_completed(0), total_trees(0), tree_progresses(nullptr) {
    omp_init_lock(&output_lock);
}

RandomForestProgress::~RandomForestProgress() {
    omp_destroy_lock(&output_lock);
    if (tree_progresses) {
        delete[] tree_progresses;
    }
}

void RandomForestProgress::initialize(int num_trees) {
    trees_completed = 0;
    total_trees = num_trees;
    
    if (tree_progresses) {
        delete[] tree_progresses;
    }
    tree_progresses = new TreeProgress[num_trees];
}

void RandomForestProgress::initialize_tree(int tree_idx, int max_depth, 
                                           int min_samples_per_leaf, size_t n_samples) {
    if (tree_idx >= 0 && tree_idx < total_trees) {
        tree_progresses[tree_idx].initialize(max_depth, min_samples_per_leaf, n_samples);
    }
}

void RandomForestProgress::increment_tree_nodes(int tree_idx) {
    if (tree_idx >= 0 && tree_idx < total_trees) {
        tree_progresses[tree_idx].increment_nodes();
    }
}

void RandomForestProgress::mark_tree_complete(int tree_idx) {
    if (tree_idx >= 0 && tree_idx < total_trees) {
        tree_progresses[tree_idx].mark_complete();
        trees_completed++;
    }
}

double RandomForestProgress::get_overall_progress() const {
    if (total_trees == 0) return 0.0;
    
    double total_progress = 0.0;
    int completed = trees_completed.load();
    
    // Add progress from completed trees
    total_progress += completed;
    
    // Add progress from in-progress trees
    for (int i = 0; i < total_trees; i++) {
        if (tree_progresses[i].nodes_created.load() > 0 && 
            tree_progresses[i].nodes_created.load() < tree_progresses[i].estimated_total_nodes) {
            total_progress += tree_progresses[i].get_progress();
        }
    }
    
    return total_progress / total_trees;
}

void RandomForestProgress::print_progress(bool force_print) {
    static int last_percent = -1;
    
    int current_percent = (int)(get_overall_progress() * 100);
    
    // Only print when percentage changes (reduce console spam) or forced
    if (force_print || current_percent != last_percent) {
        omp_set_lock(&output_lock);
        
        int completed = trees_completed.load();
        cout << "\rTraining Progress: [";
        
        // Progress bar (50 characters wide)
        int filled = (int)(get_overall_progress() * 50);
        for (int i = 0; i < 50; i++) {
            if (i < filled) cout << "█";
            else cout << "░";
        }
        
        cout << "] " << setw(3) << current_percent << "% (" 
             << completed << "/" << total_trees << " trees)" << flush;
        
        last_percent = current_percent;
        
        omp_unset_lock(&output_lock);
    }
}

void RandomForestProgress::finish() {
    omp_set_lock(&output_lock);
    cout << "\rTraining Progress: [";
    for (int i = 0; i < 50; i++) cout << "█";
    cout << "] 100% (" << total_trees << "/" << total_trees << " trees)\n";
    omp_unset_lock(&output_lock);
}

