# RANDOM FOREST 10 TREES

=== BENCHMARKING RANDOM FOREST ===
Running tests with different parallelism configurations...
Number of trees: 10
Note: Large datasets may take several minutes per configuration.

[1/3] Testing fully serial version...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 25% subset for faster training (approx 3,400 samples)
Data loaded. Fitting forest with 10 trees...

=== Results ===
Accuracy:  0.923529
Precision: 0.937468
Recall:    0.934178
F1 score:  0.93582
Training & evaluation time taken: 13237 milliseconds
[2/3] Testing with tree-level parallelism...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 25% subset for faster training (approx 3,400 samples)
Data loaded. Fitting forest with 10 trees...

=== Results ===
Accuracy:  0.923529
Precision: 0.937468
Recall:    0.934178
F1 score:  0.93582
Training & evaluation time taken: 10537 milliseconds
[3/3] Testing with forest-level parallelism...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 25% subset for faster training (approx 3,400 samples)
Data loaded. Fitting forest with 10 trees...

=== Results ===
Accuracy:  0.923529
Precision: 0.937468
Recall:    0.934178
F1 score:  0.93582
Training & evaluation time taken: 3320 milliseconds

========================================
       BENCHMARK RESULTS
========================================

Configuration                 Time (ms)      Speedup     Accuracy
---------------------------------------------------------------------
Serial (No Parallelism)       13237.00       1.00        x0.9235
Tree-level Parallelism        10537.00       1.26        x0.9235
Forest-level Parallelism      3320.00        3.99        x0.9235
---------------------------------------------------------------------