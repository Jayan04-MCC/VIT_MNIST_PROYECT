//
// Created by JAYAN on 01/07/2025.
//

#include "../../include/utils/file_io.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace FileIO {

    std::vector<std::string> split_string(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string token;
        
        while (std::getline(ss, token, delimiter)) {
            // Remove leading/trailing whitespace
            size_t start = token.find_first_not_of(" \t\r\n");
            size_t end = token.find_last_not_of(" \t\r\n");
            
            if (start != std::string::npos) {
                token = token.substr(start, end - start + 1);
                tokens.push_back(token);
            }
        }
        
        return tokens;
    }

    bool file_exists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }

    Matrix load_matrix_from_csv(const std::string& filename, bool has_header) {
        if (!file_exists(filename)) {
            throw std::runtime_error("File not found: " + filename);
        }

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::vector<std::vector<double>> data;
        std::string line;
        size_t expected_cols = 0;
        size_t line_number = 0;

        // Skip header if present
        if (has_header && std::getline(file, line)) {
            line_number++;
        }

        while (std::getline(file, line)) {
            line_number++;
            
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }

            std::vector<std::string> tokens = split_string(line, ',');
            
            if (tokens.empty()) {
                continue;
            }

            // Set expected columns from first data row
            if (data.empty()) {
                expected_cols = tokens.size();
            } else if (tokens.size() != expected_cols) {
                throw std::runtime_error("Inconsistent number of columns at line " + 
                                       std::to_string(line_number) + ". Expected " + 
                                       std::to_string(expected_cols) + ", got " + 
                                       std::to_string(tokens.size()));
            }

            std::vector<double> row;
            row.reserve(tokens.size());

            for (const std::string& token : tokens) {
                try {
                    double value = std::stod(token);
                    row.push_back(value);
                } catch (const std::exception& e) {
                    throw std::runtime_error("Invalid number format '" + token + 
                                           "' at line " + std::to_string(line_number));
                }
            }

            data.push_back(row);
        }

        file.close();

        if (data.empty()) {
            throw std::runtime_error("No data found in file: " + filename);
        }

        // Create Matrix from data
        size_t rows = data.size();
        size_t cols = data[0].size();
        Matrix result(rows, cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j];
            }
        }

        return result;
    }

    std::vector<double> load_vector_from_csv(const std::string& filename, bool has_header) {
        if (!file_exists(filename)) {
            throw std::runtime_error("File not found: " + filename);
        }

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::vector<double> data;
        std::string line;
        size_t line_number = 0;

        // Skip header if present
        if (has_header && std::getline(file, line)) {
            line_number++;
        }

        while (std::getline(file, line)) {
            line_number++;
            
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }

            std::vector<std::string> tokens = split_string(line, ',');
            
            // If multiple columns, take only the first one
            if (!tokens.empty()) {
                try {
                    double value = std::stod(tokens[0]);
                    data.push_back(value);
                } catch (const std::exception& e) {
                    throw std::runtime_error("Invalid number format '" + tokens[0] + 
                                           "' at line " + std::to_string(line_number));
                }
            }
        }

        file.close();

        if (data.empty()) {
            throw std::runtime_error("No data found in file: " + filename);
        }

        return data;
    }

    void save_matrix_to_csv(const Matrix& matrix, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create file: " + filename);
        }

        for (size_t i = 0; i < matrix.getRows(); ++i) {
            for (size_t j = 0; j < matrix.getCols(); ++j) {
                file << matrix(i, j);
                if (j < matrix.getCols() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }

        file.close();
    }

    Matrix load_vector_as_matrix(const std::string& filename, bool has_header) {
        // Load as regular vector first
        std::vector<double> vector_data = load_vector_from_csv(filename, has_header);
        
        // Create a 1xN matrix (row vector)
        Matrix result(1, vector_data.size());
        
        for (size_t i = 0; i < vector_data.size(); ++i) {
            result(0, i) = vector_data[i];
        }
        
        return result;
    }

}