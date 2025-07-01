//
// Created by JAYAN on 30/06/2025.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <initializer_list>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols, double value = 0.0);
    Matrix(const std::initializer_list<std::initializer_list<double>>& init_list);

    // Copy constructor and assignment
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    // Move constructor and assignment
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    // Destructor
    ~Matrix() = default;

    // Element access
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    // Dimensions
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    std::pair<size_t, size_t> shape() const { return {rows, cols}; }

    // Utility functions
    void fill(double value);
    void resize(size_t new_rows, size_t new_cols, double value = 0.0);

    // Display
    void print() const;

    // Static factory methods
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix identity(size_t size);
    static Matrix random(size_t rows, size_t cols, double min = 0.0, double max = 1.0);

    // Basic operators
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    // Comparison
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;

    // Friends for scalar operations
    friend Matrix operator*(double scalar, const Matrix& matrix);
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
};


#endif //MATRIX_H
