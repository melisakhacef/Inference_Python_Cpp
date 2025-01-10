cmake_minimum_required(VERSION 3.12)

# Set project name and build type
project(DNNInference)
set(CMAKE_BUILD_TYPE Release)

# Set libtorch paths (mettre à jour les chemins si nécessaire)
set(CMAKE_PREFIX_PATH "/home/meli/dnn_inference/libtorch/share/cmake/Torch")

# Find Torch
find_package(Torch REQUIRED)

# Créer un répertoire de build séparé (optionnel mais recommandé)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Ajouter les fichiers sources
add_executable(inference /home/meli/dnn_inference/test_mnist_libtorch.cpp)

# Lier les bibliothèques
target_link_libraries(inference "${TORCH_LIBRARIES}")

# Définir le standard C++
set_property(TARGET inference PROPERTY CXX_STANDARD 17)
