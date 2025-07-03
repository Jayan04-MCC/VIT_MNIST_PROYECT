//
// Created by JAYAN on 03/07/2025.
//

#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "../matrix/matrix.h"
#include "../utils/file_io.h"
#include <string>

class LayerNorm {
private:
    Matrix gamma;           // Scale parameters (weight)
    Matrix beta;            // Shift parameters (bias)
    double epsilon;         // Small constant for numerical stability
    int features;           // Number of features

public:
    // Constructor
    LayerNorm(int features, double eps = 1e-5);
    
    // Default constructor for dynamic initialization
    LayerNorm();
    
    // Forward pass
    Matrix forward(const Matrix& input);
    
    // Load weights from CSV files
    void load_weights(const std::string& base_path, int layer_idx, const std::string& norm_type);
    
    // Getters
    const Matrix& get_gamma() const { return gamma; }
    const Matrix& get_beta() const { return beta; }
    double get_epsilon() const { return epsilon; }
    int get_features() const { return features; }
    
    // Initialize with specific dimensions
    void initialize(int features, double eps = 1e-5);
};

#endif //LAYER_NORM_H