//
// Created by JAYAN on 30/06/2025.
//

#include "../../include/matrix/matrix_ops.h"
#include <cmath>
#include <algorithm>

namespace MatrixOps {

Matrix matmul(const Matrix& a, const Matrix& b) {
    if (a.getCols() != b.getRows()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    size_t rows = a.getRows();
    size_t cols = b.getCols();
    size_t inner = a.getCols();

    Matrix result(rows, cols, 0.0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner; ++k) {
                result(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    return result;
}

Matrix elementWiseMultiply(const Matrix& a, const Matrix& b) {
    if (a.getRows() != b.getRows() || a.getCols() != b.getCols()) {
        throw std::invalid_argument("Matrices must have same dimensions for element-wise multiplication");
    }

    Matrix result(a.getRows(), a.getCols());
    for (size_t i = 0; i < a.getRows(); ++i) {
        for (size_t j = 0; j < a.getCols(); ++j) {
            result(i, j) = a(i, j) * b(i, j);
        }
    }
    return result;
}

Matrix elementWiseDivide(const Matrix& a, const Matrix& b) {
    if (a.getRows() != b.getRows() || a.getCols() != b.getCols()) {
        throw std::invalid_argument("Matrices must have same dimensions for element-wise division");
    }

    Matrix result(a.getRows(), a.getCols());
    for (size_t i = 0; i < a.getRows(); ++i) {
        for (size_t j = 0; j < a.getCols(); ++j) {
            if (b(i, j) == 0.0) {
                throw std::invalid_argument("Division by zero in element-wise division");
            }
            result(i, j) = a(i, j) / b(i, j);
        }
    }
    return result;
}

Matrix transpose(const Matrix& matrix) {
    Matrix result(matrix.getCols(), matrix.getRows());
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            result(j, i) = matrix(i, j);
        }
    }
    return result;
}

Matrix addBroadcast(const Matrix& matrix, const Matrix& vector, bool row_vector) {
    Matrix result(matrix.getRows(), matrix.getCols());

    if (row_vector) {
        // Broadcasting row vector across all rows
        if (vector.getCols() != matrix.getCols() || vector.getRows() != 1) {
            throw std::invalid_argument("Vector dimensions incompatible for row broadcasting");
        }

        for (size_t i = 0; i < matrix.getRows(); ++i) {
            for (size_t j = 0; j < matrix.getCols(); ++j) {
                result(i, j) = matrix(i, j) + vector(0, j);
            }
        }
    } else {
        // Broadcasting column vector across all columns
        if (vector.getRows() != matrix.getRows() || vector.getCols() != 1) {
            throw std::invalid_argument("Vector dimensions incompatible for column broadcasting");
        }

        for (size_t i = 0; i < matrix.getRows(); ++i) {
            for (size_t j = 0; j < matrix.getCols(); ++j) {
                result(i, j) = matrix(i, j) + vector(i, 0);
            }
        }
    }

    return result;
}

Matrix multiplyBroadcast(const Matrix& matrix, const Matrix& vector, bool row_vector) {
    Matrix result(matrix.getRows(), matrix.getCols());

    if (row_vector) {
        if (vector.getCols() != matrix.getCols() || vector.getRows() != 1) {
            throw std::invalid_argument("Vector dimensions incompatible for row broadcasting");
        }

        for (size_t i = 0; i < matrix.getRows(); ++i) {
            for (size_t j = 0; j < matrix.getCols(); ++j) {
                result(i, j) = matrix(i, j) * vector(0, j);
            }
        }
    } else {
        if (vector.getRows() != matrix.getRows() || vector.getCols() != 1) {
            throw std::invalid_argument("Vector dimensions incompatible for column broadcasting");
        }

        for (size_t i = 0; i < matrix.getRows(); ++i) {
            for (size_t j = 0; j < matrix.getCols(); ++j) {
                result(i, j) = matrix(i, j) * vector(i, 0);
            }
        }
    }

    return result;
}

double sum(const Matrix& matrix) {
    double total = 0.0;
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            total += matrix(i, j);
        }
    }
    return total;
}

double mean(const Matrix& matrix) {
    return sum(matrix) / (matrix.getRows() * matrix.getCols());
}

Matrix sumAxis(const Matrix& matrix, int axis) {
    if (axis == 0) {
        // Sum across rows (result is row vector)
        Matrix result(1, matrix.getCols(), 0.0);
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            for (size_t i = 0; i < matrix.getRows(); ++i) {
                result(0, j) += matrix(i, j);
            }
        }
        return result;
    } else if (axis == 1) {
        // Sum across columns (result is column vector)
        Matrix result(matrix.getRows(), 1, 0.0);
        for (size_t i = 0; i < matrix.getRows(); ++i) {
            for (size_t j = 0; j < matrix.getCols(); ++j) {
                result(i, 0) += matrix(i, j);
            }
        }
        return result;
    } else {
        throw std::invalid_argument("Axis must be 0 or 1");
    }
}

Matrix meanAxis(const Matrix& matrix, int axis) {
    Matrix result = sumAxis(matrix, axis);
    if (axis == 0) {
        result = result / static_cast<double>(matrix.getRows());
    } else {
        result = result / static_cast<double>(matrix.getCols());
    }
    return result;
}

Matrix power(const Matrix& matrix, double exponent) {
    Matrix result(matrix.getRows(), matrix.getCols());
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            result(i, j) = std::pow(matrix(i, j), exponent);
        }
    }
    return result;
}

Matrix sqrt(const Matrix& matrix) {
    return power(matrix, 0.5);
}

Matrix exp(const Matrix& matrix) {
    Matrix result(matrix.getRows(), matrix.getCols());
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            result(i, j) = std::exp(matrix(i, j));
        }
    }
    return result;
}

Matrix log(const Matrix& matrix) {
    Matrix result(matrix.getRows(), matrix.getCols());
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            if (matrix(i, j) <= 0.0) {
                throw std::invalid_argument("Logarithm of non-positive number");
            }
            result(i, j) = std::log(matrix(i, j));
        }
    }
    return result;
}

double trace(const Matrix& matrix) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square to calculate trace");
    }

    double tr = 0.0;
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        tr += matrix(i, i);
    }
    return tr;
}

double determinant(const Matrix& matrix) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square to calculate determinant");
    }

    size_t n = matrix.getRows();

    if (n == 1) {
        return matrix(0, 0);
    } else if (n == 2) {
        return matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
    } else if (n == 3) {
        return matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) -
               matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
               matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
    } else {
        throw std::invalid_argument("Determinant calculation only implemented for matrices up to 3x3");
    }
}

Matrix inverse(const Matrix& matrix) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square to calculate inverse");
    }

    double det = determinant(matrix);
    if (std::abs(det) < 1e-10) {
        throw std::invalid_argument("Matrix is singular and cannot be inverted");
    }

    size_t n = matrix.getRows();

    if (n == 1) {
        Matrix result(1, 1);
        result(0, 0) = 1.0 / matrix(0, 0);
        return result;
    } else if (n == 2) {
        Matrix result(2, 2);
        result(0, 0) = matrix(1, 1) / det;
        result(0, 1) = -matrix(0, 1) / det;
        result(1, 0) = -matrix(1, 0) / det;
        result(1, 1) = matrix(0, 0) / det;
        return result;
    } else {
        throw std::invalid_argument("Matrix inverse only implemented for matrices up to 2x2");
    }
}

} // namespace MatrixOps