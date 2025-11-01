#ifndef LOADERS_H
#define LOADERS_H

#include <string>
#include <vector>
#include <memory>
#include <map>

using namespace std;

// Base column class
class col {
public:
    virtual unique_ptr<col> clone() const = 0;
    virtual size_t size() const = 0;
    virtual string get_type() const = 0;
    virtual ~col() {}
};

// String column implementation
class string_col : public col {
private:
    vector<string> data;
    
public:
    explicit string_col(const vector<string>& values);
    
    const string& get(size_t index) const;
    const vector<string>& get_data() const;
    
    unique_ptr<col> clone() const override;
    size_t size() const override;
    string get_type() const override;
};

// Integer column implementation
class int_col : public col {
private:
    vector<int> data;
    
public:
    explicit int_col(const vector<int>& values);
    
    int get(size_t index) const;
    const vector<int>& get_data() const;
    
    unique_ptr<col> clone() const override;
    size_t size() const override;
    string get_type() const override;
};

// Float column implementation
class float_col : public col {
private:
    vector<double> data;
    
public:
    explicit float_col(const vector<double>& values);
    
    double get(size_t index) const;
    const vector<double>& get_data() const;
    
    unique_ptr<col> clone() const override;
    size_t size() const override;
    string get_type() const override;
};

// Import CSV files and store efficiently as a dataframe
// Designed for parallel read access after initialization (training decision trees and random forests)
// Supports column data types: string, int, and float
class data_frame {
private:
    map<string, unique_ptr<col>> columns;
    vector<string> column_order;  // To maintain insertion order
    size_t num_rows = 0;
    
public:
    // Import from CSV file - returns a new data_frame
    static data_frame import_from(const string& path);
    
    // Get a column by name (returns nullptr if not found)
    const col* get_column(const string& name) const;
    
    // Get column as specific type (throws if wrong type or doesn't exist)
    const string_col* get_string_column(const string& name) const;
    const int_col* get_int_column(const string& name) const;
    const float_col* get_float_column(const string& name) const;
    
    // Get all column names
    vector<string> get_column_names() const;
    
    // Get number of rows and columns
    size_t get_num_rows() const;
    size_t get_num_columns() const;
    
    // Function to perform a train-test split
    // Returns pair: (training_data, test_data)
    pair<data_frame, data_frame> train_test_split(double test_ratio = 0.2, unsigned int seed = 42) const;
    
    // Get a subset of rows (used internally for train_test_split)
    data_frame get_rows(const vector<size_t>& indices) const;
    
    // Print basic info about the dataframe
    void print_info() const;
    
    // Test function
    void hello();
};

#endif // LOADERS_H