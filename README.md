# Parallel-random-forest

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