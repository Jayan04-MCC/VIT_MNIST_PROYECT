#include <iostream>
#include "include/matrix/matrix.h"
#include "include/matrix/matrix_ops.h"
#include "include/matrix/activation_functions.h"

int main() {
    try {
        std::cout << "=== Matrix Library Demo ===" << std::endl;

        // Test basic matrix operations
        std::cout << "\n1. Basic Matrix Operations:" << std::endl;
        Matrix a = {{1, 2, 3}, {4, 5, 6}};
        Matrix b = {{7, 8}, {9, 10}, {11, 12}};

        std::cout << "Matrix A:" << std::endl;
        a.print();
        std::cout << "\nMatrix B:" << std::endl;
        b.print();

        // Matrix multiplication
        Matrix c = MatrixOps::matmul(a, b);
        std::cout << "\nA * B:" << std::endl;
        c.print();

        // Test activation functions
        std::cout << "\n2. Activation Functions:" << std::endl;
        Matrix input = {{-2, -1, 0, 1, 2}, {-1, 0, 1, 2, 3}};
        std::cout << "Input:" << std::endl;
        input.print();

        Matrix relu_result = ActivationFunctions::relu(input);
        std::cout << "\nReLU:" << std::endl;
        relu_result.print();

        Matrix gelu_result = ActivationFunctions::gelu(input);
        std::cout << "\nGELU:" << std::endl;
        gelu_result.print();

        Matrix sigmoid_result = ActivationFunctions::sigmoid(input);
        std::cout << "\nSigmoid:" << std::endl;
        sigmoid_result.print();

        // Test softmax
        std::cout << "\n3. Softmax:" << std::endl;
        Matrix softmax_input = {{1, 2, 3}, {4, 5, 6}};
        std::cout << "Softmax Input:" << std::endl;
        softmax_input.print();

        Matrix softmax_result = ActivationFunctions::softmax(softmax_input);
        std::cout << "\nSoftmax Result:" << std::endl;
        softmax_result.print();

        // Test matrix operations
        std::cout << "\n4. Matrix Operations:" << std::endl;
        Matrix matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        std::cout << "Original Matrix:" << std::endl;
        matrix.print();

        Matrix transposed = MatrixOps::transpose(matrix);
        std::cout << "\nTransposed:" << std::endl;
        transposed.print();

        std::cout << "\nSum: " << MatrixOps::sum(matrix) << std::endl;
        std::cout << "Mean: " << MatrixOps::mean(matrix) << std::endl;

        // Test broadcasting
        std::cout << "\n5. Broadcasting:" << std::endl;
        Matrix base = {{1, 2, 3}, {4, 5, 6}};
        Matrix row_vector = {{10, 20, 30}};

        std::cout << "Base Matrix:" << std::endl;
        base.print();
        std::cout << "\nRow Vector:" << std::endl;
        row_vector.print();

        Matrix broadcast_result = MatrixOps::addBroadcast(base, row_vector, true);
        std::cout << "\nBroadcast Addition:" << std::endl;
        broadcast_result.print();

        // Test layer normalization
        std::cout << "\n6. Layer Normalization:" << std::endl;
        Matrix ln_input = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        Matrix gamma = {{1, 1, 1, 1}};
        Matrix beta = {{0, 0, 0, 0}};

        std::cout << "Input:" << std::endl;
        ln_input.print();

        Matrix ln_result = ActivationFunctions::layerNorm(ln_input, gamma, beta);
        std::cout << "\nLayer Normalized:" << std::endl;
        ln_result.print();

        // Test static factory methods
        std::cout << "\n7. Factory Methods:" << std::endl;
        Matrix zeros = Matrix::zeros(2, 3);
        std::cout << "Zeros Matrix:" << std::endl;
        zeros.print();

        Matrix identity = Matrix::identity(3);
        std::cout << "\nIdentity Matrix:" << std::endl;
        identity.print();

        Matrix random = Matrix::random(2, 3, 0.0, 1.0);
        std::cout << "\nRandom Matrix:" << std::endl;
        random.print();

        std::cout << "\n=== Demo Complete ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}