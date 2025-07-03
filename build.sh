#!/bin/bash

# Script de compilación para el proyecto VIT MNIST

echo "Compilando proyecto VIT MNIST..."

# Compilar el proyecto
g++ -o programa main.cpp \
    src/matrix/matrix.cpp \
    src/matrix/matrix_ops.cpp \
    src/matrix/activation_functions.h.cpp \
    src/utils/file_io.cpp \
    src/transformer/layer_norm.cpp \
    src/transformer/embedding.cpp \
    -Iinclude/ \
    -std=c++17 \
    -O2

# Verificar si la compilación fue exitosa
if [ $? -eq 0 ]; then
    echo "✓ Compilación exitosa!"
    echo "Ejecuta el programa con: ./programa"
else
    echo "✗ Error en la compilación"
    exit 1
fi