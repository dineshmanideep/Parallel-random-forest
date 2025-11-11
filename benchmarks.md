## Testing Dry Beans Dataset on Random Forest
=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 3,403 samples
Data loaded. Fitting forest with 100 trees...

=== Results ===
Accuracy:  0.919118
Precision: 0.936728
Recall:    0.930168
F1 score:  0.933436
Training & evaluation time taken: 130327 milliseconds
[2/3] Testing with tree-level parallelism...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 3,403 samples
Data loaded. Fitting forest with 100 trees...
Training Progress: [██████████████████████████████████████████████████] 100% (100/100 trees) 

=== Results ===
Accuracy:  0.919118
Precision: 0.936728
Recall:    0.930168
F1 score:  0.933436
Training & evaluation time taken: 100561 milliseconds
[3/3] Testing with forest-level parallelism...

=== Testing Random Forest ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 3,403 samples
Data loaded. Fitting forest with 100 trees...
Training Progress: [██████████████████████████████████████████████████] 100% (100/100 trees)

=== Results ===
Accuracy:  0.919118
Precision: 0.936728
Recall:    0.930168
F1 score:  0.933436
Training & evaluation time taken: 28140 milliseconds

========================================
       BENCHMARK RESULTS
========================================

Configuration                 Time (ms)      Speedup     Accuracy
---------------------------------------------------------------------
Serial (No Parallelism)       130327.00      1.00        x0.9191
Tree-level Parallelism        100561.00      1.30        x0.9191
Forest-level Parallelism      28140.00       4.63        x0.9191
---------------------------------------------------------------------





## Benchmarking decision tree on dry beans
=== BENCHMARKING DECISION TREE ===
Running tests with different parallelism configurations...

[1/2] Testing fully serial version...

=== Testing Decision Tree ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 3,403 samples
Data loaded and encoded. Fitting tree...
Predicting...

=== Results ===
Accuracy:  0.898529
Precision: 0.920448
Recall:    0.916834
F1 score:  0.918637
Training & evaluation time taken: 4851 milliseconds
[2/2] Testing with tree-level parallelism...

=== Testing Decision Tree ===
Loading dataset from: dataset/Dry_Bean_Dataset.csv
Using 3,403 samples
Data loaded and encoded. Fitting tree...
Predicting...

=== Results ===
Accuracy:  0.898529
Precision: 0.920448
Recall:    0.916834
F1 score:  0.918637
Training & evaluation time taken: 3313 milliseconds

========================================
       BENCHMARK RESULTS
========================================

Configuration                 Time (ms)      Speedup     Accuracy
---------------------------------------------------------------------
Serial (No Parallelism)       4851.00        1.00        x0.8985
Tree-level Parallelism        3313.00        1.46        x0.8985
---------------------------------------------------------------------

