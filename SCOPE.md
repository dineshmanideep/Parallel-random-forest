# Scope of the project / ideas etc.

Resource: https://developers.google.com/machine-learning/decision-forests

Palmer Penguins Dataset: https://allisonhorst.github.io/palmerpenguins/articles/intro.html

## What's parallel about this project
1. Parallel decision tree     (binary tree divide and conquer problem) done
2. Parallel random forests    (each tree is a thread) done
3. Parallel grid search       (training possible hyperparameters parallely to get optimal ones)
4. Parallel RF progress tracking (careful handling of output streams using omp_lock_t)

## File structurae
parallel-random-forest/
├── CMakeLists.txt
├── include/
│   ├── dataset.h                 # CSV loader, synthetic generator, train/test split
│   ├── utils.h                   # small helpers: arg parsing, RNG, timing, math
│   ├── metrics.h                 # evaluation metrics (confusion matrix, precision/recall/F1)
│   ├── node.h                    # struct Node (tree node representation)
│   ├── decision_tree.h           # DecisionTree (serial API) + config struct
│   ├── decision_tree_parallel.h  # DecisionTreeParallel (inherits/implements same API)
│   ├── random_forest.h           # RandomForest (serial API)
│   ├── random_forest_parallel.h  # RandomForestParallel (parallel tree-level + optionally internal parallel DT)
│   └── config.h                  # hyperparameters struct usable across modules
├── src/
│   ├── main.cpp                  # CLI: load data, run experiments, print metrics
│   ├── dataset.cpp
│   ├── utils.cpp
│   ├── metrics.cpp
│   ├── node.cpp                  # small (optional); can inline in header
│   ├── decision_tree.cpp
│   ├── decision_tree_parallel.cpp
│   ├── random_forest.cpp
│   └── random_forest_parallel.cpp
└── tests/
    ├── test_small.cpp            # small tests, e.g., on tiny datasets
    └── ...


Limitations
1. Only classification tasks
2. Only axis-alignend splits
3. Only binary decision trees
4. No missing values



>>> **How to parallelize a singular decision tree**

1. Feature-parallel split search (per node)

    When searching the best split at a node: evaluate each feature independently. Use #pragma omp parallel for across features.

    Each thread sorts and sweeps that feature on the subset of indices and reports the best split for that feature. Then reduce to pick the global best.

    Pros: simple, scales with number of features F. Works best when F is large.

    Cons: sorting per feature per node can be redundant; can be mitigated by caching sorted orders globally (more complex).

2. Node-parallel tree growth (breadth-first)

    Grow multiple nodes at the same tree depth in parallel. For example, perform training in level-order: find best splits for all current frontier nodes concurrently. Use OpenMP to parallelize across nodes.

    Pros: exploits parallelism as more nodes appear; avoids contention on shared data.

    Cons: needs orchestration (keeping frontier), and data duplication overhead.


Parallelizing Forests 
-> tree level parallelism
-> Output stream as a shared resource for logging progress


>>> **Metrics for classification**
1. Accuracy
2. Precision, Recall, F1 score


>>> **Eval Pipeline**
Load or generate dataset.
Split into train/test.
Train serial decision tree; measure training time.
Train parallel decision tree; measure training time; compare predictions.
Train serial RF; measure training time & accuracy.
Train parallel RF; measure training time & accuracy.
Output metrics + confusion matrices + timings.