#include <omp.h>
#include <iostream>
#include <cstdio>
#include "loaders.h"

int main() {
    int n = omp_get_num_threads();
    printf("Hello World! Threads are %d\n", n);

    data_frame df;
    df.hello();
}