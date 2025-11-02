#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <atomic>
#include <omp.h>

using namespace std;

// Progress tracker for a single decision tree
struct TreeProgress {
    atomic<int> nodes_created;
    int estimated_total_nodes;
    
    TreeProgress();
    
    // Initialize with estimate based on hyperparameters
    void initialize(int max_depth, int min_samples_per_leaf, size_t n_samples);
    
    void increment_nodes();
    
    double get_progress() const;
    
    void mark_complete();
};

// Progress tracker for random forest
struct RandomForestProgress {
    atomic<int> trees_completed;
    int total_trees;
    TreeProgress* tree_progresses;  // Array of per-tree progress
    omp_lock_t output_lock;         // Lock for thread-safe console output
    
    RandomForestProgress();
    
    ~RandomForestProgress();
    
    void initialize(int num_trees);
    
    void initialize_tree(int tree_idx, int max_depth, int min_samples_per_leaf, size_t n_samples);
    
    void increment_tree_nodes(int tree_idx);
    
    void mark_tree_complete(int tree_idx);
    
    double get_overall_progress() const;
    
    // Thread-safe progress display
    void print_progress(bool force_print = false);
    
    void finish();
};

#endif // PROGRESS_HPP