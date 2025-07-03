//
// Created by JAYAN on 03/07/2025.
//

#include "../../include/transformer/layer_norm.h"
#include "../../include/matrix/activation_functions.h"
#include "../../include/matrix/matrix_ops.h"
#include <iostream>
#include <stdexcept>

LayerNorm::LayerNorm(int features, double eps) : features(features), epsilon(eps) {
    // Initialize gamma to ones and beta to zeros
    gamma = Matrix::ones(1, features);
    beta = Matrix::zeros(1, features);
}

LayerNorm::LayerNorm() : features(0), epsilon(1e-5) {
    // Default constructor - will be initialized later
}

void LayerNorm::initialize(int features, double eps) {
    this->features = features;
    this->epsilon = eps;
    gamma = Matrix::ones(1, features);
    beta = Matrix::zeros(1, features);
}

Matrix LayerNorm::forward(const Matrix& input) {
    if (input.getCols() != features) {
        throw std::runtime_error("LayerNorm input feature dimension mismatch. Expected: " + 
                                std::to_string(features) + ", Got: " + std::to_string(input.getCols()));
    }
    
    // Use the existing layerNorm function from activation_functions
    return ActivationFunctions::layerNorm(input, gamma, beta, epsilon, 1);
}

void LayerNorm::load_weights(const std::string& base_path, int layer_idx, const std::string& norm_type) {
    try {
        std::string weight_path, bias_path;
        
        if (layer_idx == -1) {
            // Final layer norm (not implemented in current structure, but prepared for future)
            weight_path = base_path + "/norm_weight.csv";
            bias_path = base_path + "/norm_bias.csv";
        } else {
            // Transformer layer norm
            weight_path = base_path + "/transformer_layers/transformer_" + std::to_string(layer_idx) + 
                         "_" + norm_type + "_weight.csv";
            bias_path = base_path + "/transformer_layers/transformer_" + std::to_string(layer_idx) + 
                       "_" + norm_type + "_bias.csv";
        }
        
        // Load weights as matrices (CSV files have headers)
        Matrix weight_matrix = FileIO::load_matrix_from_csv(weight_path, true);
        Matrix bias_matrix = FileIO::load_matrix_from_csv(bias_path, true);
        
        // Ensure they are row vectors
        if (weight_matrix.getRows() > 1) {
            // If loaded as column vector, transpose
            weight_matrix = MatrixOps::transpose(weight_matrix);
        }
        if (bias_matrix.getRows() > 1) {
            bias_matrix = MatrixOps::transpose(bias_matrix);
        }
        
        // Set the dimensions
        features = weight_matrix.getCols();
        gamma = weight_matrix;
        beta = bias_matrix;
        
        std::cout << "LayerNorm weights loaded successfully for layer " << layer_idx 
                  << " " << norm_type << std::endl;
        std::cout << "Features: " << features << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load LayerNorm weights: " + std::string(e.what()));
    }
}