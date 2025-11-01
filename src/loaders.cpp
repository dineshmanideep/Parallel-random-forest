/*
Handles simple loading CSVs for dataset
*/

#include "loaders.h"
#include <fstream>
#include <sstream>

#include <iostream>

// Implement the methods declared in the header.
void data_frame::import_from(const std::string& path) {
    // TODO: implement CSV parsing here. For now, just avoid unused-param warnings.
    (void)path;
}

void data_frame::hello() {
    std::cout << "Hello!\n";
}