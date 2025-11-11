# RANDOM FORESTS 100 TREES WITH PROGRESS BAR (LOCKING DELAY)


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

