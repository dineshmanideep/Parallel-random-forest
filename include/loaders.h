#ifndef LOADERS_H
#define LOADERS_H

#include <string>

class data_frame {
public:
    // Declare interface only; define implementations in loaders.cpp
    static void import_from(const std::string& path);
    void hello();
};

#endif // LOADERS_H