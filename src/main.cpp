#include <omp.h>
#include <iostream>
#include <cstdio>

#include "loaders.hpp"         // Dataset handling
#include "decision_tree.hpp"   // Serial decision tree
#include "metrics.hpp"         // Metrics for evaluation

void test_decision_tree() {
    // Importing dataset
    data_frame df = data_frame::import_from("dataset/palmer_penguins.csv");

    // Train test split
    auto [train_df, test_df] = df.train_test_split(0.2);

    // Automatically fit encoding for specific string columns (only results)
    train_df.get_string_column("species")->fit_encoding();
    test_df.get_string_column("species")->fit_encoding();



    cout << "Data loaded and encoded. Fitting tree..." << endl;

    // Initializing decision tree
    decision_tree tree;

    // Setting tree growing configuration, algorithsm, etc.
    tree_growing_config growing_config;
    growing_config.criterion = tree_growing_config::SplitCriterion::GINI;
    growing_config.max_features_per_split = -1;
    tree.growing_config = &growing_config;

    // Setting hyperparameters for regularization
    tree_hyperparameters hp_config;
    hp_config.max_depth = 100;
    hp_config.min_examples_per_leaf = 5;
    tree.hp_config = &hp_config;

    // Fitting tree to training data 
    // using specified features and target feature
    tree.fit(
        train_df, 
        {"island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"}, 
        "species"
    );



    cout << "Predicting..." << endl;
    
    // Get predictions for test set
    vector<int> predictions = tree.predict(test_df);

    // Get true labels for test set
    vector<string> labels = ((string_col*) test_df.get_column("species"))->get_data();
    vector<int> encoded_labels;
    for(string& label : labels) {
        encoded_labels.push_back(
            test_df.get_string_column("species")->encode(label)
        );
    }

    // Accuracy
    cout << "Accuracy: " << metrics::accuracy(predictions, encoded_labels) << endl;
    cout << "Precision: " << metrics::precision(predictions, encoded_labels) << endl;
    cout << "Recall: " << metrics::recall(predictions, encoded_labels) << endl;
    cout << "F1 score: " << metrics::f1_score(predictions, encoded_labels) << endl;
}

int main() {
    int n = omp_get_num_threads();
    cout << "Hello World! Threads are " << n << endl;

    test_decision_tree();
}
