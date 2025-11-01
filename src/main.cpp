#include <omp.h>
#include <iostream>
#include <cstdio>

#include "loaders.hpp"
#include "decision_tree.hpp"
#include "metrics.hpp"

void test_decision_tree() {
    data_frame df = data_frame::import_from("dataset/palmer_penguins.csv");
    df.get_string_column("species")->fit_encoding();

    cout << "Data loaded and encoded. Fitting tree..." << endl;
    
    decision_tree tree;

    tree_growing_config growing_config;
    growing_config.criterion = tree_growing_config::SplitCriterion::GINI;
    growing_config.max_features_per_split = -1;
    tree.growing_config = &growing_config;

    tree.hp_config = nullptr;

    tree.fit(df, {"bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"}, "species");

    cout << "Predicting..." << endl;
    vector<int> predictions = tree.predict(df);

    vector<string> labels = ((string_col*)df.get_column("species"))->get_data();
    vector<int> encoded_labels;
    for(string& label : labels) {
        encoded_labels.push_back(
            df.get_string_column("species")->encode(label)
        );
    }

    cout << "Accuracy: " << metrics::accuracy(predictions, encoded_labels) << endl;
}

int main() {
    int n = omp_get_num_threads();
    cout << "Hello World! Threads are " << n << endl;

    test_decision_tree();
}
