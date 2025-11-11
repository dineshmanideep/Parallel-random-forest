Notes:
1. Each page of report should have a header text (on the side): 
    NITK, Surathkal - Dept. of Information Technology - IT302
2. No actual C++ code, when pseudocode is required (explicitly specified where it should appear, only 2 points)
3. Whole report must fit in 4 pages ATMOST.
4. Be concise, straight to the point and not many buzz words.
5. Each subsubsection should only be atmost 10-15 lines of content
6. Latex format

Sections

# Introduction
Introduction 

# Objectives
no need to make this point based

# Methodology
1. Decision tree introduction, formula, gini impurity, one vs rest
2. Random forests introduction, formula, bootstrapping, majority voting
3. Custom data frame implementation, no external library approach, etc. (refer to @include/loaders.hpp file)
4. Tree level parallelization - divide and conquer approach used, OMP tasks(very short pseudocode to show parallelism)
5. Random forest level parallelization (very short pseudocode to show parallelism) - parallel for, how openmp manages complexity
6. Progress bar usage - locking when updating progress (omp_lock_t example)
7. Model performance metrics we are using - accuracy, precision, recall, f1
8. Parallelism performance metrics we are using (only speedup), formula

# Benchmark Results and Graphs 

## Experimental Setup

### About Dataset
Dry beans prediction - predict type of bean based on some classifications
3848 samples
16 numerical features
categorical output feature
bootstrapping with 55% in random forests

### About Tested Machine
from @about_machine.md

## Benchmarks
(each will have some text introducing why this results are important, then a graph image, then the results)
1. Speedup discussion between varying number of trees (with progress bar) (2, 10, 100, 200 trees) 
    - talk about how in 2 trees version, parallel overhead causes serial to be faster
2. Speedup discussion between varying number of samples (100, 500, 1500, 3500)
    - talk abow how with early increase in samples, parallel overhead decreases
3. Speedup discussion between showing progress bar and not (100 trees)
    - without progress bar performs slightly better
4. Accuracy discussion between single decision tree and random forest of 200 trees
    - stark contrast

(and any other pics present in @pics/ folder)

# Conclusion
Ending notes and acknowledgements, google course thanks etc.
https://developers.google.com/machine-learning/decision-forests
(No future work and all)

# References
- google ML course on decision forests
- paper on decision trees
- openmp documentation
- early paper on amdahl's law or something, something old


