# Vision Transformer en C++ - Plan de Desarrollo de 5 Días

Eres un desarrollador experto en C++ encargado de implementar un Vision Transformer (ViT) desde cero para clasificación de MNIST. Sigue este plan estructurado de 5 días para construir una implementación completa y funcional.

## 🎯 Objetivos Generales del Proyecto
- Crear un Vision Transformer completamente funcional en C++
- Implementar todos los componentes desde cero (sin librerías de ML externas)
- Lograr resultados comparables a la implementación de referencia en PyTorch
- Enfocarse en claridad del código y corrección antes que optimización

## 📋 Estructura del Proyecto
```
vision_transformer_cpp/
├── include/
│   ├── matrix/
│   │   ├── matrix.h
│   │   ├── matrix_ops.h
│   │   └── activation_functions.h
│   ├── transformer/
│   │   ├── embedding.h
│   │   ├── layer_norm.h
│   │   ├── attention.h
│   │   ├── mlp.h
│   │   └── transformer.h
│   └── utils/
│       ├── file_io.h
│       └── image_processing.h
├── src/
├── tests/
└── data/
    └── weights_csv_organized/
```

---

## 📅 DÍA 1: Fundamentos - Librería de Matrices (6-8 horas)
**Prioridad: Construir una base matemática sólida**

### Sesión Matutina (3-4 horas): Clase Matrix Principal
Crear `include/matrix/matrix.h` y `src/matrix/matrix.cpp`:

**Requisitos Esenciales:**
- Matriz 2D dinámica con asignación en heap
- Constructor: `Matrix(int rows, int cols)`
- Acceso a elementos: `operator()(int i, int j)` y `at(int i, int j)`
- Manejo de memoria: destructor apropiado y constructor de copia
- Métodos de inicialización: `zeros()`, `ones()`, `random()`
- Getters de dimensiones: `rows()`, `cols()`, `size()`
- Método de debug: `print()` con formato

**Notas Críticas de Implementación:**
- Usar almacenamiento row-major (`data[i*cols + j]`)
- Implementar verificación de límites con assertions
- Manejar fallos de asignación de memoria elegantemente

### Sesión Vespertina (3-4 horas): Operaciones de Matrices
Crear `include/matrix/matrix_ops.h` y `src/matrix/matrix_ops.cpp`:

**Operaciones Principales:**
- Multiplicación de matrices: `matmul(const Matrix& A, const Matrix& B)`
- Operaciones elemento a elemento: `add()`, `subtract()`, `multiply()`
- Operaciones de forma: `transpose()`, `reshape()`
- Broadcasting: soporte básico para operaciones escalares

**Consideraciones de Rendimiento:**
- Optimizar multiplicación de matrices con loop tiling
- Agregar verificaciones de compatibilidad de dimensiones
- Usar referencias `const` para parámetros de entrada

### Sesión Nocturna (1 hora): Sistema de E/S de Archivos
Crear `include/utils/file_io.h` y `src/utils/file_io.cpp`:

**Funciones de Carga de Archivos:**
- `load_matrix_from_csv(const string& filename)`
- `load_vector_from_csv(const string& filename)`
- Manejo de errores para archivos mal formados
- Soporte para diferentes delimitadores

### 🎯 Criterios de Éxito del Día 1:
- [ ] La clase Matrix pasa todas las pruebas de operaciones básicas
- [ ] La multiplicación de matrices funciona correctamente para varios tamaños
- [ ] La carga de CSV funciona con archivos de datos de muestra
- [ ] Sin fugas de memoria (verificar con valgrind si está disponible)

---

## 📅 DÍA 2: Componentes de Red Neuronal (6-8 horas)
**Prioridad: Implementar funciones de activación y normalización**

### Sesión Matutina (2-3 horas): Funciones de Activación
Crear `include/matrix/activation_functions.h` y `src/matrix/activation_functions.cpp`:

**Funciones Requeridas:**
- `softmax(const Matrix& input)` - con estabilidad numérica
- `gelu(const Matrix& input)` - Gaussian Error Linear Unit
- `relu(const Matrix& input)` - para propósitos de debugging

**Implementación de GELU:**
```cpp
// GELU(x) = x * Φ(x) donde Φ es CDF de la normal estándar
// Aproximación: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Estabilidad de Softmax:**
- Restar valor máximo antes del exponencial
- Manejar casos extremos (todos ceros, valores muy grandes)

### Sesión Vespertina (3-4 horas): Normalización de Capas
Crear `include/transformer/layer_norm.h` y `src/transformer/layer_norm.cpp`:

**Funcionalidad Principal:**
- Método `forward(const Matrix& input)`
- Parámetros: vectores `gamma` (escala) y `beta` (desplazamiento)
- Calcular media y varianza a través de la dimensión de características
- Normalizar: `(x - media) / sqrt(varianza + epsilon)`
- Aplicar parámetros aprendidos: `gamma * normalizado + beta`

**Detalles de Implementación:**
- Epsilon por defecto: 1e-5 para estabilidad numérica
- `load_weights(const string& base_path)` - carga desde tu estructura CSV:
  - Para capas transformer: `base_path + "/transformer_layers/blocks_X_norm1_weight.csv"`
  - Para norm final: `base_path + "/layer_norm/norm_weight.csv"`
- Soporte para diferentes dimensiones de entrada

**Ejemplo de Carga de Pesos:**
```cpp
void LayerNorm::load_weights(const string& base_path, int layer_idx, const string& norm_type) {
    string weight_path = base_path + "/transformer_layers/blocks_" + 
                        std::to_string(layer_idx) + "_" + norm_type + "_weight.csv";
    string bias_path = base_path + "/transformer_layers/blocks_" + 
                      std::to_string(layer_idx) + "_" + norm_type + "_bias.csv";
    gamma = load_vector_from_csv(weight_path);
    beta = load_vector_from_csv(bias_path);
}
```

### Sesión Nocturna (2 horas): Capa de Embedding
Crear `include/transformer/embedding.h` y `src/transformer/embedding.cpp`:

**Patch Embedding:**
- Convertir parches de imagen a vectores de características
- Proyección lineal: `parches * matriz_peso + bias`
- Agregar token de clase aprendible al inicio
- Agregar embeddings posicionales

**Métodos Clave:**
- `forward(const Matrix& image_patches)`
- `load_weights(const string& base_path)` - carga desde tu estructura CSV:
  - `base_path + "/patch_embedding/patch_proj_weight.csv"`
  - `base_path + "/patch_embedding/patch_proj_bias.csv"`
  - `base_path + "/position_embedding/pos_embed.csv"`
  - `base_path + "/class_token/cls_token.csv"`
- Manejar diferentes tamaños de parches y dimensiones de imagen

**Ejemplo de Carga de Pesos:**
```cpp
void PatchEmbedding::load_weights(const string& base_path) {
    proj_weight = load_matrix_from_csv(base_path + "/patch_embedding/patch_proj_weight.csv");
    proj_bias = load_vector_from_csv(base_path + "/patch_embedding/patch_proj_bias.csv");
    pos_embed = load_matrix_from_csv(base_path + "/position_embedding/pos_embed.csv");
    cls_token = load_matrix_from_csv(base_path + "/class_token/cls_token.csv");
}
```

### 🎯 Criterios de Éxito del Día 2:
- [ ] Todas las funciones de activación producen salidas esperadas
- [ ] La normalización de capas coincide con la implementación de PyTorch
- [ ] La capa de embedding maneja el procesamiento de parches correctamente
- [ ] La carga de pesos funciona desde archivos CSV

---

## 📅 DÍA 3: Multi-Head Self-Attention (8-10 horas)
**Prioridad: Implementar el mecanismo central del transformer**

### Sesión Matutina (4-5 horas): Self-Attention Básico
Crear `include/transformer/attention.h` y `src/transformer/attention.cpp`:

**Pasos del Mecanismo de Atención:**
1. **Proyecciones Lineales:** Generar matrices Q, K, V
   ```cpp
   Q = input * W_q + b_q
   K = input * W_k + b_k  
   V = input * W_v + b_v
   ```

2. **Atención Escalada Dot-Product:**
   ```cpp
   scores = Q * K^T / sqrt(d_k)
   attention_weights = softmax(scores)
   output = attention_weights * V
   ```

**Detalles Críticos de Implementación:**
- Manejar correctamente la longitud de secuencia y dimensiones de características
- Implementar enmascaramiento de atención (si es necesario)
- Agregar estabilidad numérica al softmax en atención

### Sesión Vespertina (4-5 horas): Multi-Head Attention
**Procesamiento Multi-Head:**
1. **Reshape para Múltiples Cabezas:**
   - Dividir Q, K, V en `num_heads` piezas
   - Cada cabeza procesa `d_model / num_heads` dimensiones

2. **Atención Paralela:**
   - Aplicar mecanismo de atención a cada cabeza independientemente
   - Mantener pesos de atención separados por cabeza

3. **Concatenar y Proyectar:**
   - Combinar todas las salidas de cabezas
   - Aplicar proyección lineal final

**Métodos Clave:**
- `forward(const Matrix& input)`
- `load_weights(const string& base_path, int layer_idx)` - carga desde tu estructura CSV:
  - `base_path + "/transformer_layers/blocks_X_attn_qkv_weight.csv"`
  - `base_path + "/transformer_layers/blocks_X_attn_qkv_bias.csv"`
  - `base_path + "/transformer_layers/blocks_X_attn_proj_weight.csv"`
  - `base_path + "/transformer_layers/blocks_X_attn_proj_bias.csv"`
- `get_attention_weights()` para visualización

**Ejemplo de Carga de Pesos:**
```cpp
void MultiHeadAttention::load_weights(const string& base_path, int layer_idx) {
    string prefix = base_path + "/transformer_layers/blocks_" + std::to_string(layer_idx) + "_attn_";
    qkv_weight = load_matrix_from_csv(prefix + "qkv_weight.csv");
    qkv_bias = load_vector_from_csv(prefix + "qkv_bias.csv");
    proj_weight = load_matrix_from_csv(prefix + "proj_weight.csv");
    proj_bias = load_vector_from_csv(prefix + "proj_bias.csv");
}
```

### ⚠️ Desafíos Críticos del Día 3:
- **Manejo de Dimensiones:** Rastrear formas cuidadosamente (batch_size, seq_len, d_model)
- **Eficiencia de Memoria:** Evitar copias innecesarias durante el reshape
- **Debugging:** Implementar visualización de pesos de atención

### 🎯 Criterios de Éxito del Día 3:
- [ ] La atención de una sola cabeza produce formas de salida correctas
- [ ] La atención multi-cabeza combina cabezas apropiadamente
- [ ] Los pesos de atención suman 1.0 a través de la dimensión de secuencia
- [ ] El rendimiento es aceptable para longitudes de secuencia objetivo

---

## 📅 DÍA 4: MLP y Integración de Capas Transformer (6-8 horas)
**Prioridad: Completar todos los componentes del transformer**

### Sesión Matutina (2-3 horas): Red Feed-Forward MLP
Crear `include/transformer/mlp.h` y `src/transformer/mlp.cpp`:

**Arquitectura MLP:**
```cpp
class MLPBlock {
    Matrix fc1_weight, fc1_bias;  // Primera capa lineal
    Matrix fc2_weight, fc2_bias;  // Segunda capa lineal
    
    Matrix forward(const Matrix& input) {
        auto hidden = gelu(input * fc1_weight + fc1_bias);
        return hidden * fc2_weight + fc2_bias;
    }
    
    void load_weights(const string& base_path, int layer_idx) {
        string prefix = base_path + "/transformer_layers/blocks_" + std::to_string(layer_idx) + "_mlp_";
        fc1_weight = load_matrix_from_csv(prefix + "fc1_weight.csv");
        fc1_bias = load_vector_from_csv(prefix + "fc1_bias.csv");
        fc2_weight = load_matrix_from_csv(prefix + "fc2_weight.csv");
        fc2_bias = load_vector_from_csv(prefix + "fc2_bias.csv");
    }
};
```

**Dimensiones Típicas:**
- Entrada: `[seq_len, d_model]`
- Oculta: `[seq_len, 4 * d_model]` (factor de expansión de 4)
- Salida: `[seq_len, d_model]`

### Sesión Vespertina (3-4 horas): Capa Transformer Completa
Crear capa transformer completa en `include/transformer/transformer.h`:

**Estructura del Bloque Transformer:**
```cpp
Matrix transformer_layer(const Matrix& input) {
    // 1. Pre-normalización + Multi-Head Attention + Residual
    auto norm1_out = layer_norm1.forward(input);
    auto attn_out = multi_head_attention.forward(norm1_out);
    auto residual1 = input + attn_out;  // Conexión residual
    
    // 2. Pre-normalización + MLP + Residual  
    auto norm2_out = layer_norm2.forward(residual1);
    auto mlp_out = mlp.forward(norm2_out);
    return residual1 + mlp_out;  // Conexión residual
}
```

### Sesión Nocturna (2 horas): Stack de Múltiples Capas
**Apilamiento de Capas:**
- Implementar bucle sobre N capas transformer
- Mantener pesos separados para cada capa
- Manejar flujo de gradiente a través de conexiones residuales

**Gestión de Pesos:**
- Organizar pesos por capa usando tu estructura CSV:
  - `weights_csv_organized/transformer_layers/blocks_0_*`
  - `weights_csv_organized/transformer_layers/blocks_1_*`
  - etc.
- Implementar carga eficiente de pesos:

```cpp
void TransformerStack::load_all_weights(const string& base_path, int num_layers) {
    for (int i = 0; i < num_layers; i++) {
        layers[i].attention.load_weights(base_path, i);
        layers[i].mlp.load_weights(base_path, i);
        layers[i].norm1.load_weights(base_path, i, "norm1");
        layers[i].norm2.load_weights(base_path, i, "norm2");
    }
}
```

### 🎯 Criterios de Éxito del Día 4:
- [ ] MLP completamente funcional
- [ ] Una capa transformer completa funcionando
- [ ] Stack de múltiples capas
- [ ] Conexiones residuales correctas

---

## 📅 DÍA 5: Integración Final y Testing (6-8 horas)
**Prioridad: Modelo completo funcional con validación**

### Sesión Matutina (3-4 horas): Vision Transformer Completo
Finalizar `include/transformer/transformer.h` y `src/transformer/transformer.cpp`:

**Pipeline Completo de ViT:**
```cpp
class VisionTransformer {
    Matrix forward(const Matrix& image) {
        // 1. Patch embedding + codificación posicional
        auto embedded = patch_embed.forward(image);
        
        // 2. Agregar class token
        auto with_cls = add_class_token(embedded);
        
        // 3. Pasar a través de capas transformer
        auto encoded = transformer_stack.forward(with_cls);
        
        // 4. Cabeza de clasificación (usar class token)
        auto cls_token = encoded.get_row(0);  // Primer token
        return classification_head.forward(cls_token);
    }
    
    void load_all_weights(const string& weights_base_path) {
        patch_embed.load_weights(weights_base_path);
        transformer_stack.load_all_weights(weights_base_path, num_layers);
        
        // Cargar clasificador final
        Matrix head_weight = load_matrix_from_csv(weights_base_path + "/classifier/head_weight.csv");
        Matrix head_bias = load_vector_from_csv(weights_base_path + "/classifier/head_bias.csv");
        classification_head.set_weights(head_weight, head_bias);
        
        // Cargar layer norm final
        final_norm.load_weights(weights_base_path + "/layer_norm/", -1, "norm");
    }
};
```

**Procesamiento de Imágenes:**
- Convertir MNIST 28x28 a parches (ej., parches 4x4 = 49 parches)
- Aplanar parches a vectores
- Agregar embeddings posicionales

### Sesión Vespertina (2-3 horas): Aplicación Principal
Crear `src/main.cpp` y utilidades de soporte:

**Características de la Aplicación Principal:**
- Cargar imágenes de prueba MNIST
- Ejecutar inferencia en imágenes de muestra
- Mostrar predicciones con puntuaciones de confianza
- Medir tiempo de inferencia

**Utilidades de Procesamiento de Imágenes:**
- `load_mnist_image(const string& filename)`
- `preprocess_image(const Matrix& image)`
- `postprocess_predictions(const Matrix& logits)`

### Sesión Nocturna (2 horas): Testing y Validación
Crear pruebas comprehensivas en `tests/test_complete_model.cpp`:

**Pruebas de Validación:**
- **Prueba End-to-End:** Ejecutar inferencia en muestras conocidas de MNIST
- **Comparación con PyTorch:** Comparar salidas intermedias con referencia
- **Prueba de Rendimiento:** Medir tiempo de inferencia y uso de memoria
- **Prueba de Precisión:** Validar precisión de clasificación en conjunto de prueba

**Herramientas de Debugging:**
- Guardar activaciones intermedias en CSV
- Implementar visualización de atención
- Agregar profilers de tiempo para cada componente

### 🎯 Criterios de Éxito del Día 5:
- [ ] El modelo completo ejecuta inferencia exitosamente
- [ ] Las predicciones coinciden con la referencia de PyTorch (dentro de tolerancia)
- [ ] El tiempo de inferencia es razonable (< 1 segundo por imagen)
- [ ] El uso de memoria es estable (sin fugas)

---

## 🚀 Mejores Prácticas de Implementación

### Directrices de Calidad de Código:
1. **Usar RAII:** Gestión apropiada de recursos en todas las clases
2. **Correctitud de Const:** Marcar parámetros de solo lectura como `const`
3. **Manejo de Errores:** Verificar dimensiones y manejar casos extremos
4. **Documentación:** Comentar operaciones matemáticas complejas
5. **Testing:** Escribir pruebas unitarias para cada componente

### Estrategias de Debugging:
1. **Verificación de Forma:** Imprimir dimensiones de matrices en cada paso
2. **Inspección de Valores:** Guardar resultados intermedios en archivos
3. **Comparación de Referencia:** Comparar con salidas de PyTorch
4. **Testing Incremental:** Probar cada componente en aislamiento

### Consejos de Rendimiento:
1. **Layout de Memoria:** Usar orden row-major consistentemente
2. **Eficiencia de Cache:** Acceder memoria secuencialmente cuando sea posible
3. **Evitar Copias:** Usar referencias y semántica de movimiento
4. **Perfilar Primero:** Medir antes de optimizar

## ⚡ Planes de Contingencia

### Si Te Atrasas:
- **Día 1-2:** Usar librería Eigen para operaciones de matrices
- **Día 3:** Implementar solo atención de una cabeza
- **Día 4:** Probar con menos capas transformer
- **Día 5:** Enfocarse en funcionalidad básica sobre optimización

### Si Te Adelantas:
- Agregar paralelización con OpenMP
- Implementar procesamiento por lotes
- Agregar testing más comprehensivo
- Optimizar uso de memoria y rendimiento

## 📊 Métricas de Éxito
- **Funcionalidad:** El modelo produce predicciones razonables de MNIST
- **Precisión:** Dentro del 5% de la implementación de referencia de PyTorch  
- **Rendimiento:** Inferencia bajo 1 segundo por imagen
- **Calidad de Código:** Sin fugas de memoria, arquitectura limpia

¡Recuerda: Enfócate en corrección primero, optimización segundo. Una implementación que funciona pero es lenta es infinitamente mejor que una implementación rápida que no funciona!