#include <omp.h>
#include <iostream>
#include <cstdio>
#include <chrono>

#include "loaders.hpp"         // Dataset handling
#include "decision_tree.hpp"   // Decision tree
#include "metrics.hpp"         // Metrics for evaluation
#include "random_forest.hpp"   // Random forest
#include "progress.hpp"        // Progress tracking

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


    // Setting tree growing configuration, condition scores, parallel properties, etc.
    tree_growing_config growing_config;
    growing_config.criterion = tree_growing_config::SplitCriterion::GINI;
    growing_config.max_features_per_split = -1; // for random feature sampling for random forests

    growing_config.use_parallel = false;
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
    cout << "Accuracy:  " << metrics::accuracy(predictions, encoded_labels) << endl;
    cout << "Precision: " << metrics::precision(predictions, encoded_labels) << endl;
    cout << "Recall:    " << metrics::recall(predictions, encoded_labels) << endl;
    cout << "F1 score:  " << metrics::f1_score(predictions, encoded_labels) << endl;


    auto time_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << "Training & evaluation time taken: " << duration.count() << " milliseconds" << endl;
}

void test_random_forest() {

    vector<string> feature_cols = {
        "Pregnancies", 
        "Glucose", 
        "BloodPressure", 
        "SkinThickness", 
        "Insulin", 
        "BMI", 
        "DiabetesPedigreeFunction", 
        "Age"
    };
    string target_col = "Outcome";

    // Importing dataset
    data_frame df = data_frame::import_from("dataset/diabetes.csv");

    auto [train_df, test_df] = df.train_test_split(0.2);
    // train_df.get_string_column(target_col)->fit_encoding();
    // test_df.get_string_column(target_col)->fit_encoding();

    cout << "Data loaded and encoded. Fitting forest..." << endl;

    // Initializing random forest
    random_forest forest;

    // Setting random forest configuration
    random_forest_config config;
    config.num_trees = 5000;
    config.bootstrap_sample_ratio = 0.55;
    config.use_parallel = true;

    forest.rf_config = &config;

    // Setting tree growing configuration, condition scores, parallel properties, etc.
    tree_growing_config growing_config;
    growing_config.criterion = tree_growing_config::SplitCriterion::GINI;
    growing_config.max_features_per_split = -1; // for random feature sampling for random forests

    growing_config.use_parallel = false;
    growing_config.min_samples_for_parallel = 100;
    growing_config.max_parallel_depth = 8;

    forest.growing_config = &growing_config;

    // Setting hyperparameters for regularization
    tree_hyperparameters hp_config;
    hp_config.max_depth = 300;
    hp_config.min_examples_per_leaf = 20;

    forest.hp_config = &hp_config;

    // Enable progress tracking
    RandomForestProgress progress;
    forest.progress_tracker = &progress;

    // Fitting forest to training data
    auto time_start = chrono::high_resolution_clock::now();
    forest.fit(train_df, feature_cols, target_col);

    // Getting predictions for test set
    vector<int> predictions = forest.predict(test_df);
    // Getting true labels for test set
    //vector<string> labels = ((string_col*) test_df.get_column(target_col))->get_data();
    // vector<int> encoded_labels;
    // for(string& label : labels) {
    //     encoded_labels.push_back(test_df.get_string_column(target_col)->encode(label));
    // }
    vector<int> encoded_labels = test_df.get_int_column(target_col)->get_data();

    // Accuracy
    cout << "Accuracy:  " << metrics::accuracy(predictions, encoded_labels) << endl;
    cout << "Precision: " << metrics::precision(predictions, encoded_labels) << endl;
    cout << "Recall:    " << metrics::recall(predictions, encoded_labels) << endl;
    cout << "F1 score:  " << metrics::f1_score(predictions, encoded_labels) << endl;

    auto time_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << "Training & evaluation time taken: " << duration.count() << " milliseconds" << endl;
}

#include <sstream>
#include <semaphore>

int main() {
    
    // Testing random forest with progress tracking
    test_random_forest();

    // Testing decision tree
    // test_decision_tree();

    return 0;
}
