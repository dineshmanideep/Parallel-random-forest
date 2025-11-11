# RANDOM FOREST 200 TREES

=== BENCHMARKING RANDOM FOREST ===
Running tests with different parallelism configurations...
Number of trees: 200
Note: Large datasets may take several minutes per configuration.

[1/3] Testing fully serial version...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 25% subset for faster training (approx 3,400 samples)
Data loaded. Fitting forest with 200 trees...
Training Progress: [██████████████████████████████████████████████████] 100% (200/200 trees)

=== Results ===
Accuracy:  0.919118
Precision: 0.936728
Recall:    0.930168
F1 score:  0.933436
Training & evaluation time taken: 262589 milliseconds
[2/3] Testing with tree-level parallelism...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 25% subset for faster training (approx 3,400 samples)
Data loaded. Fitting forest with 200 trees...
Training Progress: [██████████████████████████████████████████████████] 100% (200/200 trees)

=== Results ===
Accuracy:  0.919118
Precision: 0.936728
Recall:    0.930168
F1 score:  0.933436
Training & evaluation time taken: 200831 milliseconds
[3/3] Testing with forest-level parallelism...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 25% subset for faster training (approx 3,400 samples)
Data loaded. Fitting forest with 200 trees...
Training Progress: [██████████████████████████████████████████████████] 100% (200/200 trees)

=== Results ===
Accuracy:  0.919118
Precision: 0.936728
Recall:    0.930168
F1 score:  0.933436
Training & evaluation time taken: 58047 milliseconds

========================================
       BENCHMARK RESULTS
========================================

Configuration                 Time (ms)      Speedup     Accuracy
---------------------------------------------------------------------
Serial (No Parallelism)       262589.00      1.00        x0.9191
Tree-level Parallelism        200831.00      1.31        x0.9191
Forest-level Parallelism      58047.00       4.52        x0.9191
---------------------------------------------------------------------


========================================
   Complete!
========================================