//
// Created by JAYAN on 30/06/2025.
//

#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include "matrix.h"

namespace MatrixOps {
    // Matrix multiplication
    Matrix matmul(const Matrix& a, const Matrix& b);

    // Element-wise operations
    Matrix elementWiseMultiply(const Matrix& a, const Matrix& b);
    Matrix elementWiseDivide(const Matrix& a, const Matrix& b);

    // Matrix operations
    Matrix transpose(const Matrix& matrix);

    // Broadcasting operations
    Matrix addBroadcast(const Matrix& matrix, const Matrix& vector, bool row_vector = true);
    Matrix multiplyBroadcast(const Matrix& matrix, const Matrix& vector, bool row_vector = true);

    // Reduction operations
    double sum(const Matrix& matrix);
    double mean(const Matrix& matrix);
    Matrix sumAxis(const Matrix& matrix, int axis); // axis 0: sum columns, axis 1: sum rows
    Matrix meanAxis(const Matrix& matrix, int axis);

    // Utility functions
    Matrix power(const Matrix& matrix, double exponent);
    Matrix sqrt(const Matrix& matrix);
    Matrix exp(const Matrix& matrix);
    Matrix log(const Matrix& matrix);

    // Matrix properties
    double trace(const Matrix& matrix);
    double determinant(const Matrix& matrix); // For small matrices
    Matrix inverse(const Matrix& matrix); // For small matrices
}


#endif //MATRIX_OPS_H
