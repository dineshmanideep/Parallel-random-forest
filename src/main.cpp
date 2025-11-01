#include <omp.h>
#include <iostream>
#include <cstdio>
#include "loaders.h"

int main() {
    int n = omp_get_num_threads();
    cout << "Hello World! Threads are " << n << endl;

    data_frame df = data_frame::import_from("dataset/test.csv");
    cout << df.get_string_column("TestString")->get_data()[1] << endl;
}