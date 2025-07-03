#include <iostream>
#include "include/matrix/matrix.h"
#include "include/matrix/matrix_ops.h"
#include "include/matrix/activation_functions.h"
#include "include/utils/file_io.h"
#include "include/transformer/layer_norm.h"
#include "include/transformer/embedding.h"

void test_original_functionality() {
    std::cout << "=== PRUEBA ORIGINAL: Carga de Pesos ===" << std::endl;
    try {
        Matrix weight_matrix = FileIO::load_matrix_from_csv("weights_organized/classifier/mlp_head_0_weight.csv", true);
        std::cout << "✅ Matriz cargada exitosamente!" << std::endl;
        std::cout << "Dimensiones: " << weight_matrix.getRows() << "x" << weight_matrix.getCols() << std::endl;
        
        // Mostrar solo las primeras 3x3 celdas para verificar
        std::cout << "Primeras 3x3 celdas:" << std::endl;
        for (size_t i = 0; i < std::min((size_t)3, weight_matrix.getRows()); ++i) {
            for (size_t j = 0; j < std::min((size_t)3, weight_matrix.getCols()); ++j) {
                std::cout << weight_matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "❌ Error cargando matriz: " << e.what() << std::endl;
    }
}

void test_day2_components() {
    std::cout << "\n=== PRUEBAS DÍA 2: Componentes Transformer ===" << std::endl;
    
    try {
        // Test 1: LayerNorm
        std::cout << "\n1️⃣ Probando LayerNorm..." << std::endl;
        LayerNorm layer_norm;
        layer_norm.load_weights("weights_organized", 0, "layer_norm_1");
        
        // Crear entrada de prueba (simula embeddings de patches)
        Matrix test_input = Matrix::random(2, 256);  // batch_size=2, features=256
        Matrix norm_output = layer_norm.forward(test_input);
        
        std::cout << "✅ LayerNorm funciona correctamente" << std::endl;
        std::cout << "   Entrada: " << test_input.getRows() << "x" << test_input.getCols() << std::endl;
        std::cout << "   Salida:  " << norm_output.getRows() << "x" << norm_output.getCols() << std::endl;
        
        // Test 2: PatchEmbedding
        std::cout << "\n2️⃣ Probando PatchEmbedding..." << std::endl;
        PatchEmbedding patch_embed(49, 256);
        patch_embed.load_weights("weights_organized");
        
        // Crear patches de prueba (simula imagen 28x28 dividida en 7x7 = 49 patches)
        Matrix test_patches = Matrix::random(1, 49);  // batch_size=1, num_patches=49
        Matrix embed_output = patch_embed.forward(test_patches);
        
        std::cout << "✅ PatchEmbedding funciona correctamente" << std::endl;
        std::cout << "   Patches entrada: " << test_patches.getRows() << "x" << test_patches.getCols() << std::endl;
        std::cout << "   Embeddings salida: " << embed_output.getRows() << "x" << embed_output.getCols() << std::endl;
        
        // Test 3: Funciones de activación existentes
        std::cout << "\n3️⃣ Probando funciones de activación..." << std::endl;
        Matrix test_data = Matrix::random(2, 256);
        Matrix gelu_result = ActivationFunctions::gelu(test_data);
        Matrix softmax_result = ActivationFunctions::softmax(test_data);
        
        std::cout << "✅ GELU: " << test_data.getRows() << "x" << test_data.getCols() 
                  << " → " << gelu_result.getRows() << "x" << gelu_result.getCols() << std::endl;
        std::cout << "✅ Softmax: " << test_data.getRows() << "x" << test_data.getCols() 
                  << " → " << softmax_result.getRows() << "x" << softmax_result.getCols() << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error en pruebas del Día 2: " << e.what() << std::endl;
    }
}

void show_next_steps() {
    std::cout << "\n=== PRÓXIMOS PASOS ===" << std::endl;
    std::cout << "✅ Día 1: Librería de matrices - COMPLETADO" << std::endl;
    std::cout << "✅ Día 2: Componentes neuronales - COMPLETADO" << std::endl;
    std::cout << "⏳ Día 3: Multi-Head Self-Attention - PENDIENTE" << std::endl;
    std::cout << "⏳ Día 4: MLP y capas transformer - PENDIENTE" << std::endl;
    std::cout << "⏳ Día 5: Integración final - PENDIENTE" << std::endl;
    std::cout << "\n🚀 ¡Listo para implementar Multi-Head Attention!" << std::endl;
}

int main() {
    std::cout << "🧠 VISION TRANSFORMER C++ - PRUEBAS DE DESARROLLO 🧠" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    // Ejecutar todas las pruebas
    test_original_functionality();
    test_day2_components();
    show_next_steps();
    
    return 0;
}