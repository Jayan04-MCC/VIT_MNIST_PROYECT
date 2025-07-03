//
// Created by JAYAN on 03/07/2025.
//

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "../matrix/matrix.h"
#include "../utils/file_io.h"
#include <string>

class PatchEmbedding {
private:
    Matrix proj_weight;         // Projection weight matrix (features, num_patches)
    Matrix proj_bias;           // Projection bias vector (features,)
    Matrix pos_embed;           // Positional embeddings (seq_len, features)
    Matrix cls_token;           // Class token (1, features)
    
    int num_patches;            // Number of patches (e.g., 49 for 7x7 patches)
    int features;               // Feature dimension (e.g., 256)
    int seq_len;                // Sequence length (num_patches + 1 for class token)

public:
    // Constructor
    PatchEmbedding(int num_patches, int features = 256);
    
    // Default constructor
    PatchEmbedding();
    
    // Forward pass: convert image patches to embeddings
    Matrix forward(const Matrix& image_patches);
    
    // Load weights from CSV files
    void load_weights(const std::string& base_path);
    
    // Utility functions
    Matrix add_class_token(const Matrix& embedded_patches);
    Matrix add_positional_embeddings(const Matrix& embedded_with_cls);
    
    // Getters
    const Matrix& get_proj_weight() const { return proj_weight; }
    const Matrix& get_proj_bias() const { return proj_bias; }
    const Matrix& get_pos_embed() const { return pos_embed; }
    const Matrix& get_cls_token() const { return cls_token; }
    int get_num_patches() const { return num_patches; }
    int get_features() const { return features; }
    int get_seq_len() const { return seq_len; }
    
    // Initialize with specific dimensions
    void initialize(int num_patches, int features);
};

#endif //EMBEDDING_H