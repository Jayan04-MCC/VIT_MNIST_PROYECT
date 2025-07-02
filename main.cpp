#include <iostream>
#include "include/matrix/matrix.h"
#include "include/matrix/matrix_ops.h"
#include "include/matrix/activation_functions.h"
#include "include/utils/file_io.h"

int main() {
        try {
            Matrix weight_matrix = FileIO::load_matrix_from_csv("weights_organized/classifier/mlp_head_0_weight.csv", true);
            std::cout << "Matriz cargada exitosamente!" << std::endl;
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
            std::cout << "Error cargando matriz: " << e.what() << std::endl;
        }
    return 0;
}