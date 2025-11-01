/*
Handles simple loading CSVs for dataset
*/

#include "loaders.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <cctype>

using namespace std;

// ==================== String Column Implementation ====================

string_col::string_col(const vector<string>& values) : data(values) {}

void string_col::add_value(const string& value) {
    data.push_back(value);
}

const string& string_col::get(size_t index) const {
    if (index >= data.size()) {
        throw out_of_range("Index out of range in string_col");
    }
    return data[index];
}

const vector<string>& string_col::get_data() const {
    return data;
}

unique_ptr<col> string_col::clone() const {
    return make_unique<string_col>(data);
}

size_t string_col::size() const {
    return data.size();
}

string string_col::get_type() const {
    return "string";
}

// ==================== Integer Column Implementation ====================

int_col::int_col(const vector<int>& values) : data(values) {}

void int_col::add_value(int value) {
    data.push_back(value);
}

int int_col::get(size_t index) const {
    if (index >= data.size()) {
        throw out_of_range("Index out of range in int_col");
    }
    return data[index];
}

const vector<int>& int_col::get_data() const {
    return data;
}

unique_ptr<col> int_col::clone() const {
    return make_unique<int_col>(data);
}

size_t int_col::size() const {
    return data.size();
}

string int_col::get_type() const {
    return "int";
}

// ==================== Float Column Implementation ====================

float_col::float_col(const vector<double>& values) : data(values) {}

void float_col::add_value(double value) {
    data.push_back(value);
}

double float_col::get(size_t index) const {
    if (index >= data.size()) {
        throw out_of_range("Index out of range in float_col");
    }
    return data[index];
}

const vector<double>& float_col::get_data() const {
    return data;
}

unique_ptr<col> float_col::clone() const {
    return make_unique<float_col>(data);
}

size_t float_col::size() const {
    return data.size();
}

string float_col::get_type() const {
    return "float";
}

// ==================== Helper Functions ====================

// Trim whitespace from both ends of a string
static string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

// Parse a CSV line considering quoted fields
static vector<string> parse_csv_line(const string& line) {
    vector<string> fields;
    string field;
    bool in_quotes = false;
    
    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];
        
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            fields.push_back(trim(field));
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(trim(field));
    
    return fields;
}

// Determine if a string is an integer
static bool is_integer(const string& str) {
    if (str.empty()) return false;
    
    size_t start = 0;
    if (str[0] == '-' || str[0] == '+') {
        if (str.length() == 1) return false;
        start = 1;
    }
    
    for (size_t i = start; i < str.length(); ++i) {
        if (!isdigit(str[i])) return false;
    }
    return true;
}

// Determine if a string is a float
static bool is_float(const string& str) {
    if (str.empty()) return false;
    
    char* end;
    strtod(str.c_str(), &end);
    return end != str.c_str() && *end == '\0';
}

// Infer the type of a column from sample values
static string infer_type(const vector<string>& values) {
    if (values.empty()) return "string";
    
    bool could_be_int = true;
    bool could_be_float = true;
    
    for (const auto& val : values) {
        if (val.empty()) continue; // Skip empty values for type inference
        
        if (could_be_int && !is_integer(val)) {
            could_be_int = false;
        }
        if (could_be_float && !is_float(val)) {
            could_be_float = false;
        }
        
        if (!could_be_int && !could_be_float) {
            return "string";
        }
    }
    
    if (could_be_int) return "int";
    if (could_be_float) return "float";
    return "string";
}

// ==================== Data Frame Implementation ====================

data_frame::data_frame() : num_rows(0) {}

// Move constructor
data_frame::data_frame(data_frame&& other) noexcept 
    : columns(std::move(other.columns)),
      column_order(std::move(other.column_order)),
      num_rows(other.num_rows) {
    // Note: mutex cannot be moved, but a new one is default-constructed
    other.num_rows = 0;
}

// Move assignment operator
data_frame& data_frame::operator=(data_frame&& other) noexcept {
    if (this != &other) {
        columns = std::move(other.columns);
        column_order = std::move(other.column_order);
        num_rows = other.num_rows;
        // Note: mutex cannot be moved, but we keep the existing one
        other.num_rows = 0;
    }
    return *this;
}

data_frame data_frame::import_from(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + path);
    }
    
    data_frame df;
    string line;
    
    // Read header
    if (!getline(file, line)) {
        throw runtime_error("Empty file or no header: " + path);
    }
    
    vector<string> headers = parse_csv_line(line);
    size_t num_cols = headers.size();
    
    // Store all rows temporarily to infer types
    vector<vector<string>> all_rows;
    
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        vector<string> row = parse_csv_line(line);
        if (row.size() != num_cols) {
            cerr << "Warning: Skipping row with " << row.size() 
                      << " columns (expected " << num_cols << ")\n";
            continue;
        }
        all_rows.push_back(row);
    }
    
    file.close();
    
    if (all_rows.empty()) {
        cout << "Warning: No data rows found in file\n";
        df.num_rows = 0;
        return df;
    }
    
    df.num_rows = all_rows.size();
    
    // Infer type for each column and create appropriate column objects
    for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
        vector<string> column_values;
        for (const auto& row : all_rows) {
            column_values.push_back(row[col_idx]);
        }
        
        string col_type = infer_type(column_values);
        
        if (col_type == "int") {
            auto int_column = make_unique<int_col>();
            for (const auto& val : column_values) {
                if (val.empty()) {
                    int_column->add_value(0); // Default value for empty
                } else {
                    int_column->add_value(stoi(val));
                }
            }
            df.add_column(headers[col_idx], move(int_column));
            
        } else if (col_type == "float") {
            auto float_column = make_unique<float_col>();
            for (const auto& val : column_values) {
                if (val.empty()) {
                    float_column->add_value(0.0); // Default value for empty
                } else {
                    float_column->add_value(stod(val));
                }
            }
            df.add_column(headers[col_idx], move(float_column));
            
        } else {
            auto string_column = make_unique<string_col>();
            for (const auto& val : column_values) {
                string_column->add_value(val);
            }
            df.add_column(headers[col_idx], move(string_column));
        }
    }
    
    return df;
}

void data_frame::add_column(const string& name, unique_ptr<col> column) {
    if (num_rows == 0) {
        num_rows = column->size();
    } else if (column->size() != num_rows) {
        throw invalid_argument("Column size mismatch: expected " + 
                                    to_string(num_rows) + " rows");
    }
    
    column_order.push_back(name);
    columns[name] = move(column);
}

const col* data_frame::get_column(const string& name) const {
    auto it = columns.find(name);
    if (it == columns.end()) {
        return nullptr;
    }
    return it->second.get();
}

const string_col* data_frame::get_string_column(const string& name) const {
    const col* column = get_column(name);
    if (!column) {
        throw invalid_argument("Column not found: " + name);
    }
    
    const string_col* str_col = dynamic_cast<const string_col*>(column);
    if (!str_col) {
        throw invalid_argument("Column " + name + " is not a string column");
    }
    return str_col;
}

const int_col* data_frame::get_int_column(const string& name) const {
    const col* column = get_column(name);
    if (!column) {
        throw invalid_argument("Column not found: " + name);
    }
    
    const int_col* i_col = dynamic_cast<const int_col*>(column);
    if (!i_col) {
        throw invalid_argument("Column " + name + " is not an int column");
    }
    return i_col;
}

const float_col* data_frame::get_float_column(const string& name) const {
    const col* column = get_column(name);
    if (!column) {
        throw invalid_argument("Column not found: " + name);
    }
    
    const float_col* f_col = dynamic_cast<const float_col*>(column);
    if (!f_col) {
        throw invalid_argument("Column " + name + " is not a float column");
    }
    return f_col;
}

vector<string> data_frame::get_column_names() const {
    return column_order;
}

size_t data_frame::get_num_rows() const {
    return num_rows;
}

size_t data_frame::get_num_columns() const {
    return columns.size();
}

pair<data_frame, data_frame> data_frame::train_test_split(double test_ratio, unsigned int seed) const {
    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        throw invalid_argument("test_ratio must be between 0 and 1");
    }
    
    // Create shuffled indices
    vector<size_t> indices(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        indices[i] = i;
    }
    
    mt19937 rng(seed);
    shuffle(indices.begin(), indices.end(), rng);
    
    // Split indices
    size_t test_size = static_cast<size_t>(num_rows * test_ratio);
    size_t train_size = num_rows - test_size;
    
    vector<size_t> train_indices(indices.begin(), indices.begin() + train_size);
    vector<size_t> test_indices(indices.begin() + train_size, indices.end());
    
    return make_pair(get_rows(train_indices), get_rows(test_indices));
}

data_frame data_frame::copy() const {
    data_frame new_df;
    new_df.num_rows = num_rows;
    
    for (const auto& col_name : column_order) {
        auto it = columns.find(col_name);
        if (it != columns.end()) {
            new_df.column_order.push_back(col_name);
            new_df.columns[col_name] = it->second->clone();
        }
    }
    
    return new_df;
}

data_frame data_frame::get_rows(const vector<size_t>& indices) const {
    data_frame subset;
    subset.num_rows = indices.size();
    
    for (const auto& col_name : column_order) {
        auto it = columns.find(col_name);
        if (it == columns.end()) continue;
        
        const col* column = it->second.get();
        
        // Handle different column types
        if (auto str_col = dynamic_cast<const string_col*>(column)) {
            vector<string> values;
            const auto& data = str_col->get_data();
            for (size_t idx : indices) {
                if (idx < data.size()) {
                    values.push_back(data[idx]);
                }
            }
            subset.column_order.push_back(col_name);
            subset.columns[col_name] = make_unique<string_col>(values);
            
        } else if (auto i_col = dynamic_cast<const int_col*>(column)) {
            vector<int> values;
            const auto& data = i_col->get_data();
            for (size_t idx : indices) {
                if (idx < data.size()) {
                    values.push_back(data[idx]);
                }
            }
            subset.column_order.push_back(col_name);
            subset.columns[col_name] = make_unique<int_col>(values);
            
        } else if (auto f_col = dynamic_cast<const float_col*>(column)) {
            vector<double> values;
            const auto& data = f_col->get_data();
            for (size_t idx : indices) {
                if (idx < data.size()) {
                    values.push_back(data[idx]);
                }
            }
            subset.column_order.push_back(col_name);
            subset.columns[col_name] = make_unique<float_col>(values);
        }
    }
    
    return subset;
}

void data_frame::print_info() const {
    cout << "Data Frame Info:\n";
    cout << "  Rows: " << num_rows << "\n";
    cout << "  Columns: " << columns.size() << "\n";
    cout << "\nColumn Details:\n";
    
    for (const auto& col_name : column_order) {
        auto it = columns.find(col_name);
        if (it != columns.end()) {
            cout << "  - " << col_name << " (" << it->second->get_type() << ")\n";
        }
    }
}

void data_frame::hello() {
    cout << "Hello from data_frame!\n";
}