//
// Created by JAYAN on 03/07/2025.
//

#include "../../include/transformer/embedding.h"
#include "../../include/matrix/matrix_ops.h"
#include <iostream>
#include <stdexcept>

PatchEmbedding::PatchEmbedding(int num_patches, int features) 
    : num_patches(num_patches), features(features), seq_len(num_patches + 1) {
    // Initialize matrices with appropriate dimensions
    proj_weight = Matrix::zeros(features, num_patches);
    proj_bias = Matrix::zeros(1, features);
    pos_embed = Matrix::zeros(seq_len, features);
    cls_token = Matrix::zeros(1, features);
}

PatchEmbedding::PatchEmbedding() : num_patches(0), features(0), seq_len(0) {
    // Default constructor - will be initialized later
}

void PatchEmbedding::initialize(int num_patches, int features) {
    this->num_patches = num_patches;
    this->features = features;
    this->seq_len = num_patches + 1;
    
    proj_weight = Matrix::zeros(features, num_patches);
    proj_bias = Matrix::zeros(1, features);
    pos_embed = Matrix::zeros(seq_len, features);
    cls_token = Matrix::zeros(1, features);
}

Matrix PatchEmbedding::forward(const Matrix& image_patches) {
    if (image_patches.getCols() != num_patches) {
        throw std::runtime_error("PatchEmbedding input patch dimension mismatch. Expected: " + 
                                std::to_string(num_patches) + ", Got: " + std::to_string(image_patches.getCols()));
    }
    
    // Step 1: Project patches to embedding space
    // patches: (batch_size, num_patches) -> (batch_size, features)
    Matrix embedded = MatrixOps::matmul(image_patches, MatrixOps::transpose(proj_weight));
    
    // Add bias (broadcasting)
    for (int i = 0; i < embedded.getRows(); ++i) {
        for (int j = 0; j < embedded.getCols(); ++j) {
            embedded(i, j) += proj_bias(0, j);
        }
    }
    
    // Step 2: Add class token
    Matrix with_cls = add_class_token(embedded);
    
    // Step 3: Add positional embeddings
    Matrix final_embedding = add_positional_embeddings(with_cls);
    
    return final_embedding;
}

Matrix PatchEmbedding::add_class_token(const Matrix& embedded_patches) {
    int batch_size = embedded_patches.getRows();
    Matrix with_cls(batch_size, seq_len * features);
    
    // Add class token to the beginning of each sequence
    for (int b = 0; b < batch_size; ++b) {
        // Copy class token to positions 0 to features-1
        for (int f = 0; f < features; ++f) {
            with_cls(b, f) = cls_token(0, f);
        }
        
        // Copy embedded patches to positions features to seq_len*features-1
        for (int f = 0; f < features; ++f) {
            with_cls(b, features + f) = embedded_patches(b, f);
        }
    }
    
    return with_cls;
}

Matrix PatchEmbedding::add_positional_embeddings(const Matrix& embedded_with_cls) {
    Matrix result = embedded_with_cls;
    int batch_size = result.getRows();
    
    // Add positional embeddings (broadcasting across batch dimension)
    // Simplified: just add a small positional bias (skip for now to avoid dimension issues)
    // TODO: Implement proper positional embedding handling
    for (int b = 0; b < batch_size; ++b) {
        for (int f = 0; f < std::min(features, (int)result.getCols()); ++f) {
            // Add a small positional bias for now
            result(b, f) += 0.01 * f; // Simple position-dependent bias
        }
    }
    
    return result;
}

void PatchEmbedding::load_weights(const std::string& base_path) {
    try {
        // Load projection weights and bias
        std::string proj_weight_path = base_path + "/other/input_layer_weight.csv";
        std::string proj_bias_path = base_path + "/other/input_layer_bias.csv";
        
        proj_weight = FileIO::load_matrix_from_csv(proj_weight_path, true);
        proj_bias = FileIO::load_matrix_from_csv(proj_bias_path, true);
        
        // Ensure proj_bias is a row vector
        if (proj_bias.getRows() > 1) {
            proj_bias = MatrixOps::transpose(proj_bias);
        }
        
        // Load positional embeddings
        std::string pos_embed_path = base_path + "/position_embedding/pos_embedding.csv";
        pos_embed = FileIO::load_matrix_from_csv(pos_embed_path, true);
        
        // Load class token
        std::string cls_token_path = base_path + "/class_token/cls_token.csv";
        cls_token = FileIO::load_matrix_from_csv(cls_token_path, true);
        
        // Update dimensions based on loaded weights
        features = proj_weight.getRows();
        num_patches = proj_weight.getCols();
        seq_len = num_patches + 1; // Simple calculation: patches + class token
        
        std::cout << "PatchEmbedding weights loaded successfully!" << std::endl;
        std::cout << "Features: " << features << ", Patches: " << num_patches 
                  << ", Sequence Length: " << seq_len << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load PatchEmbedding weights: " + std::string(e.what()));
    }
}