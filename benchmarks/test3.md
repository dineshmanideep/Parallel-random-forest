# RANDOM FOREST 100 TREES NO PROGRESS BAR

=== BENCHMARKING RANDOM FOREST ===
Running tests with different parallelism configurations...
Number of trees: 100
Note: Large datasets may take several minutes per configuration.

[1/3] Testing fully serial version...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 25% subset for faster training (approx 3,400 samples)
Data loaded. Fitting forest with 100 trees...

=== Results ===
Accuracy:  0.919118
Precision: 0.936728
Recall:    0.930168
F1 score:  0.933436
Training & evaluation time taken: 132386 milliseconds
[2/3] Testing with tree-level parallelism...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 25% subset for faster training (approx 3,400 samples)
Data loaded. Fitting forest with 100 trees...

=== Results ===
Accuracy:  0.919118
Precision: 0.936728
Recall:    0.930168
F1 score:  0.933436
Training & evaluation time taken: 102267 milliseconds
[3/3] Testing with forest-level parallelism...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 25% subset for faster training (approx 3,400 samples)
Data loaded. Fitting forest with 100 trees...

=== Results ===
Accuracy:  0.919118
Precision: 0.936728
Recall:    0.930168
F1 score:  0.933436
Training & evaluation time taken: 29406 milliseconds

========================================
       BENCHMARK RESULTS
========================================

Configuration                 Time (ms)      Speedup     Accuracy
---------------------------------------------------------------------
Serial (No Parallelism)       132386.00      1.00        x0.9191
Tree-level Parallelism        102267.00      1.29        x0.9191
Forest-level Parallelism      29406.00       4.50        x0.9191
---------------------------------------------------------------------
