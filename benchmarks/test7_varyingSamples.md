# RANDOM FOREST 100 TREES, VARYING SAMPLES (100, 500, 1500, 3500)

========================================
       BENCHMARK RESULTS
========================================

Configuration                 Time (ms)      Speedup     Accuracy
---------------------------------------------------------------------
100 samples (Serial)          206.00         1.00        x0.1000      
100 samples (Forest-parallel) 76.00          2.71        x0.1000
500 samples (Serial)          4027.00        1.00        x0.8900
500 samples (Forest-parallel) 806.00         5.00        x0.8900
1500 samples (Serial)         28187.00       1.00        x0.9200
1500 samples (Forest-parallel)6236.00        4.52        x0.9200
3500 samples (Serial)         138875.00      1.00        x0.9057
3500 samples (Forest-parallel)28653.00       4.85        x0.9057
---------------------------------------------------------------------


=== Sample Size Scaling Analysis ===

Serial Performance:
Sample Size | Time (ms) | Time Ratio
---------------------------------------------
100         | 206.00    | 1.00      x
500         | 4027.00   | 19.55     x
1500        | 28187.00  | 136.83    x
3500        | 138875.00 | 674.15    x


Parallel Performance & Speedup:
Sample Size | Time (ms) | Speedup vs Serial
--------------------------------------------------
100         | 76.00     | 2.71             x
500         | 806.00    | 5.00             x
1500        | 6236.00   | 4.52             x
3500        | 28653.00  | 4.85             x
