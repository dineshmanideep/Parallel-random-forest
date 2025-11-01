# Parallel-random-forest

Parallel random forests and decision trees implementaiton to study parallelization speedup of these processes.

A decision tree is a supervised learning model used for classification or regression.
It recursively splits data based on features to reduce impurity (i.e., make child nodes more homogeneous).

---

## Dependencies

```bash

sudo apt-get install libomp-dev   # for OpenMP C++ SDK
sudo apt-get install cmake        # for building

```

## Running Instructions

```bash

cmake -S . -B build  # build Makefiles for the compiler
cmake --build build  # build the actual app
./build/bin/forests  # run the app

```