#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include "loaders.hpp"

using namespace std;

class metrics {
public:
    /*
        Splitter scores (impurity measures for classification)
    */
    
    // Gini impurity: measures how often a randomly chosen element would be incorrectly labeled
    // Returns value in [0, 1], where 0 = pure (all same class), higher = more mixed
    static double gini_impurity(
        const vector<int>& labels, 
        int num_classes
    );
    
    // Shannon entropy: measures information content/uncertainty in the labels
    // Returns value in [0, log2(num_classes)], where 0 = pure
    static double shannon_entropy(
        const vector<int>& labels, 
        int num_classes
    );
    
    // Gini gain: reduction in Gini impurity from a split
    // Higher is better (more information gain)
    static double gini_gain(
        const vector<int>& parent_labels,
        const vector<int>& left_labels,
        const vector<int>& right_labels,
        int num_classes
    );
    
    // Entropy gain: reduction in entropy from a split
    // Higher is better (more information gain)
    static double entropy_gain(
        const vector<int>& parent_labels,
        const vector<int>& left_labels,
        const vector<int>& right_labels,
        int num_classes
    );
    
    // Helper: compute class distribution (count of each class)
    // Returns vector of size num_classes where result[i] = count of class i
    static vector<int> class_counts(
        const vector<int>& labels, 
        int num_classes
    );
    
    /* 
        Performance metrics
    */
    
    static double accuracy(
        const vector<int>& predictions, 
        const vector<int>& labels
    );

    static double precision(
        const vector<int>& predictions, 
        const vector<int>& labels
    );

    static double recall(
        const vector<int>& predictions, 
        const vector<int>& labels
    );

    static double f1_score(
        const vector<int>& predictions, 
        const vector<int>& labels
    );
};

#endif // METRICS_H