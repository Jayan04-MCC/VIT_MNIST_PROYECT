//
// Created by JAYAN on 30/06/2025.
//

#include "../../include/matrix/matrix.h"
#include <random>
#include <iomanip>

// Default constructor
Matrix::Matrix() : rows(0), cols(0) {}

// Parameterized constructor
Matrix::Matrix(size_t rows, size_t cols, double value)
    : rows(rows), cols(cols), data(rows, std::vector<double>(cols, value)) {}

// Initializer list constructor
Matrix::Matrix(const std::initializer_list<std::initializer_list<double>>& init_list) {
    rows = init_list.size();
    if (rows == 0) {
        cols = 0;
        return;
    }

    cols = init_list.begin()->size();
    data.resize(rows);

    size_t i = 0;
    for (const auto& row : init_list) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        data[i].assign(row.begin(), row.end());
        ++i;
    }
}

// Copy constructor
Matrix::Matrix(const Matrix& other)
    : rows(other.rows), cols(other.cols), data(other.data) {}

// Copy assignment
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

// Move constructor
Matrix::Matrix(Matrix&& other) noexcept
    : rows(other.rows), cols(other.cols), data(std::move(other.data)) {
    other.rows = 0;
    other.cols = 0;
}

// Move assignment
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = std::move(other.data);
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

// Element access
double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
}

// Utility functions
void Matrix::fill(double value) {
    for (auto& row : data) {
        std::fill(row.begin(), row.end(), value);
    }
}

void Matrix::resize(size_t new_rows, size_t new_cols, double value) {
    rows = new_rows;
    cols = new_cols;
    data.assign(rows, std::vector<double>(cols, value));
}

void Matrix::print() const {
    for (const auto& row : data) {
        for (const auto& elem : row) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << elem << " ";
        }
        std::cout << std::endl;
    }
}

// Static factory methods
Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::identity(size_t size) {
    Matrix result(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

Matrix Matrix::random(size_t rows, size_t cols, double min, double max) {
    Matrix result(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = dis(gen);
        }
    }
    return result;
}

// Basic operators
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    return (*this) * (1.0 / scalar);
}

// Comparison operators
bool Matrix::operator==(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        return false;
    }

    const double epsilon = 1e-9;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (std::abs(data[i][j] - other.data[i][j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::operator!=(const Matrix& other) const {
    return !(*this == other);
}

// Friend functions
Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            os << std::setw(8) << std::fixed << std::setprecision(3) << matrix.data[i][j];
            if (j < matrix.cols - 1) os << " ";
        }
        if (i < matrix.rows - 1) os << "\n";
    }
    return os;
}