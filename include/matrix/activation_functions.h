//
// Created by JAYAN on 30/06/2025.
//

#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include "matrix.h"

namespace ActivationFunctions {

    // ReLU activation function
    Matrix relu(const Matrix& input);
    Matrix reluDerivative(const Matrix& input);

    // GELU activation function
    Matrix gelu(const Matrix& input);
    Matrix geluDerivative(const Matrix& input);

    // Softmax activation function
    Matrix softmax(const Matrix& input, int axis = 1);

    // Dropout (for inference, acts as identity)
    Matrix dropout(const Matrix& input, double dropout_rate = 0.0, bool training = false);

    // Layer normalization helpers
    Matrix layerNorm(const Matrix& input, const Matrix& gamma, const Matrix& beta,
                     double epsilon = 1e-5, int axis = 1);

    // Helper functions for layer normalization
    Matrix computeLayerNormStats(const Matrix& input, int axis = 1);
    std::pair<Matrix, Matrix> computeMeanAndVariance(const Matrix& input, int axis = 1);

    // Additional activation functions
    Matrix sigmoid(const Matrix& input);
    Matrix tanh(const Matrix& input);
    Matrix leakyRelu(const Matrix& input, double alpha = 0.01);

    // Utility functions
    Matrix clip(const Matrix& input, double min_val, double max_val);
}

#endif //ACTIVATION_FUNCTIONS_H
