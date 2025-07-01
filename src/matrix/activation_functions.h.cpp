//
// Created by JAYAN on 30/06/2025.
//

#include "../../include/matrix/activation_functions.h"
#include "../../include/matrix/matrix_ops.h"
const double M_PI = 3.14159265358979323846;
#include <cmath>
#include  <random>
#include <algorithm>

namespace ActivationFunctions {

Matrix relu(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());
    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < input.getCols(); ++j) {
            result(i, j) = std::max(0.0, input(i, j));
        }
    }
    return result;
}

Matrix reluDerivative(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());
    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < input.getCols(); ++j) {
            result(i, j) = input(i, j) > 0.0 ? 1.0 : 0.0;
        }
    }
    return result;
}

Matrix gelu(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());
    const double sqrt_2_pi = std::sqrt(2.0 / M_PI);

    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < input.getCols(); ++j) {
            double x = input(i, j);
            double tanh_arg = sqrt_2_pi * (x + 0.044715 * x * x * x);
            result(i, j) = 0.5 * x * (1.0 + std::tanh(tanh_arg));
        }
    }
    return result;
}

Matrix geluDerivative(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());
    const double sqrt_2_pi = std::sqrt(2.0 / M_PI);

    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < input.getCols(); ++j) {
            double x = input(i, j);
            double tanh_arg = sqrt_2_pi * (x + 0.044715 * x * x * x);
            double tanh_val = std::tanh(tanh_arg);
            double sech2_val = 1.0 - tanh_val * tanh_val;

            double derivative = 0.5 * (1.0 + tanh_val) +
                               0.5 * x * sech2_val * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x);
            result(i, j) = derivative;
        }
    }
    return result;
}

Matrix softmax(const Matrix& input, int axis) {
    Matrix result(input.getRows(), input.getCols());

    if (axis == 1) {
        // Softmax across columns (each row sums to 1)
        for (size_t i = 0; i < input.getRows(); ++i) {
            // Find max for numerical stability
            double max_val = input(i, 0);
            for (size_t j = 1; j < input.getCols(); ++j) {
                max_val = std::max(max_val, input(i, j));
            }

            // Compute exponentials and sum
            double sum_exp = 0.0;
            std::vector<double> exp_vals(input.getCols());
            for (size_t j = 0; j < input.getCols(); ++j) {
                exp_vals[j] = std::exp(input(i, j) - max_val);
                sum_exp += exp_vals[j];
            }

            // Normalize
            for (size_t j = 0; j < input.getCols(); ++j) {
                result(i, j) = exp_vals[j] / sum_exp;
            }
        }
    } else if (axis == 0) {
        // Softmax across rows (each column sums to 1)
        for (size_t j = 0; j < input.getCols(); ++j) {
            // Find max for numerical stability
            double max_val = input(0, j);
            for (size_t i = 1; i < input.getRows(); ++i) {
                max_val = std::max(max_val, input(i, j));
            }

            // Compute exponentials and sum
            double sum_exp = 0.0;
            std::vector<double> exp_vals(input.getRows());
            for (size_t i = 0; i < input.getRows(); ++i) {
                exp_vals[i] = std::exp(input(i, j) - max_val);
                sum_exp += exp_vals[i];
            }

            // Normalize
            for (size_t i = 0; i < input.getRows(); ++i) {
                result(i, j) = exp_vals[i] / sum_exp;
            }
        }
    } else {
        throw std::invalid_argument("Axis must be 0 or 1");
    }

    return result;
}

Matrix dropout(const Matrix& input, double dropout_rate, bool training) {
    if (!training) {
        // During inference, dropout acts as identity
        return input;
    }

    // During training, randomly set elements to zero
    Matrix result = input;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dis(1.0 - dropout_rate);

    double scale = 1.0 / (1.0 - dropout_rate);

    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < input.getCols(); ++j) {
            if (dis(gen)) {
                result(i, j) *= scale;
            } else {
                result(i, j) = 0.0;
            }
        }
    }

    return result;
}

std::pair<Matrix, Matrix> computeMeanAndVariance(const Matrix& input, int axis) {
    Matrix mean = MatrixOps::meanAxis(input, axis);

    Matrix variance;
    if (axis == 1) {
        // Compute variance across columns
        variance = Matrix(input.getRows(), 1, 0.0);
        for (size_t i = 0; i < input.getRows(); ++i) {
            double var_sum = 0.0;
            for (size_t j = 0; j < input.getCols(); ++j) {
                double diff = input(i, j) - mean(i, 0);
                var_sum += diff * diff;
            }
            variance(i, 0) = var_sum / input.getCols();
        }
    } else if (axis == 0) {
        // Compute variance across rows
        variance = Matrix(1, input.getCols(), 0.0);
        for (size_t j = 0; j < input.getCols(); ++j) {
            double var_sum = 0.0;
            for (size_t i = 0; i < input.getRows(); ++i) {
                double diff = input(i, j) - mean(0, j);
                var_sum += diff * diff;
            }
            variance(0, j) = var_sum / input.getRows();
        }
    } else {
        throw std::invalid_argument("Axis must be 0 or 1");
    }

    return {mean, variance};
}

Matrix layerNorm(const Matrix& input, const Matrix& gamma, const Matrix& beta,
                 double epsilon, int axis) {
    auto [mean, variance] = computeMeanAndVariance(input, axis);

    Matrix result(input.getRows(), input.getCols());

    if (axis == 1) {
        // Normalize across columns
        for (size_t i = 0; i < input.getRows(); ++i) {
            double std_dev = std::sqrt(variance(i, 0) + epsilon);
            for (size_t j = 0; j < input.getCols(); ++j) {
                double normalized = (input(i, j) - mean(i, 0)) / std_dev;
                result(i, j) = gamma(0, j) * normalized + beta(0, j);
            }
        }
    } else if (axis == 0) {
        // Normalize across rows
        for (size_t j = 0; j < input.getCols(); ++j) {
            double std_dev = std::sqrt(variance(0, j) + epsilon);
            for (size_t i = 0; i < input.getRows(); ++i) {
                double normalized = (input(i, j) - mean(0, j)) / std_dev;
                result(i, j) = gamma(i, 0) * normalized + beta(i, 0);
            }
        }
    }

    return result;
}

Matrix sigmoid(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());
    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < input.getCols(); ++j) {
            result(i, j) = 1.0 / (1.0 + std::exp(-input(i, j)));
        }
    }
    return result;
}

Matrix tanh(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());
    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < input.getCols(); ++j) {
            result(i, j) = std::tanh(input(i, j));
        }
    }
    return result;
}

Matrix leakyRelu(const Matrix& input, double alpha) {
    Matrix result(input.getRows(), input.getCols());
    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < input.getCols(); ++j) {
            result(i, j) = input(i, j) > 0.0 ? input(i, j) : alpha * input(i, j);
        }
    }
    return result;
}

Matrix clip(const Matrix& input, double min_val, double max_val) {
    Matrix result(input.getRows(), input.getCols());
    for (size_t i = 0; i < input.getRows(); ++i) {
        for (size_t j = 0; j < input.getCols(); ++j) {
            result(i, j) = std::clamp(input(i, j), min_val, max_val);
        }
    }
    return result;
}

} // namespace Act